import os
import pandas as pd
import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
import joblib

ModelDict = os.path.exists('./Model')
if not ModelDict:
    os.makedirs('./Model')
ResPath = os.path.exists('./res')
if not ResPath:
    os.makedirs('./res')

# str类型的是时间
curr_time = time.strftime("%Y-%m-%d", time.localtime())
# 读取数据
trainset = pd.read_csv('./Features/TrainFeatureSelected.csv', index_col=[0])
testset = pd.read_csv('./Features/EvalFeatureSelected.csv', index_col=[0])
# trainset label=1的
trainset_with_label = pd.DataFrame(trainset[trainset['ebb_eligible'] == 1])
# trainset 没有label的
trainset_without_label = pd.DataFrame(trainset[trainset['ebb_eligible'] != 1])
positive_label, negative_label = 1, -1

# Step 1. identifying reliable negative examples ########################################################################
trainset_with_label_id = trainset_with_label['customer_id'].tolist()
trainset_without_label_id = trainset_without_label['customer_id'].tolist()
trainset_with_label_label = trainset_with_label['ebb_eligible'].tolist()
trainset_without_label_label = trainset_without_label['ebb_eligible'].tolist()
del trainset_with_label['customer_id']
del trainset_without_label['customer_id']
del trainset_with_label['ebb_eligible']
del trainset_without_label['ebb_eligible']
print('Shape')
print('trainset_with_label:', trainset_with_label.shape)
print('trainset_without_label:', trainset_without_label.shape, '\n')

# 1.1将未分类的数据全部默认为-1
trainset_with_label_copy = trainset_with_label.copy(deep=True)
trainset_without_label_copy = trainset_without_label.copy(deep=True)
trainset_without_label_copy['ebb_eligible'] = negative_label
trainset_with_label_copy['ebb_eligible'] = positive_label

model_for_solid_negative = IsolationForest()
Full_labeled_SET = pd.concat([trainset_with_label_copy, trainset_without_label_copy])
y = Full_labeled_SET['ebb_eligible']
del Full_labeled_SET['ebb_eligible']
X = Full_labeled_SET
# 1.2使用一个模型训练出classifer
model_for_solid_negative.fit(X, y)

# 对unlabel的trainset_without_label进行predict_proba
X_test = trainset_without_label
# 1.3用得到的classifier对未标记数据进行predict_prob
# 输入样本的异常得分。越低，说明越不正常。 #########################################################################
y_predict = model_for_solid_negative.score_samples(X_test)
predict_prob_res = pd.DataFrame()
predict_prob_res['customer_id'] = trainset_without_label_id
predict_prob_res['score_samples'] = y_predict  # y_predict[:,0]是label=1的概率， y_predict[:,1]是label=0的概率
# 1.4 对未标记数据的predict_prob排序，选出predict_prob最小的init_num个作为solid negative
predict_prob_res = predict_prob_res.sort_values(by='score_samples', ascending=True)
quantile_df = predict_prob_res['score_samples'].quantile(
    [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]).tolist()
print('分位数\n', quantile_df)  # score_samples 由小到大， 越小说明越不正常
init_score_samples = quantile_df[0]  # 0-1之间predict_proba > init_prob_threshold的，认为更可能是label=0的
print('init_score_samples:', init_score_samples)
predict_prob_res = predict_prob_res[predict_prob_res['score_samples'] < init_score_samples]

trainset_with_label['customer_id'] = trainset_with_label_id
trainset_without_label['customer_id'] = trainset_without_label_id
trainset_with_label['ebb_eligible'] = trainset_with_label_label
trainset_without_label['ebb_eligible'] = trainset_without_label_label
# # # 1.3.1取出label=0的数据 trainset_with_label_zero
trainset_with_label_zero = pd.merge(trainset_without_label, predict_prob_res, on='customer_id')
del trainset_with_label_zero['score_samples']
trainset_with_label_zero['ebb_eligible'] = negative_label
print('trainset_with_label_zero', trainset_with_label_zero.shape)
# 1.3.2删除label=0的数据，从trainset_without_label，使用差集
trainset_without_label = pd.concat([trainset_without_label, trainset_with_label_zero]).drop_duplicates(
    subset=['customer_id'], keep=False)
print('trainset_without_label', trainset_without_label.shape)
# 1.3.3添加label=0的数据，到trainset_with_label
trainset_with_label = pd.concat([trainset_with_label, trainset_with_label_zero])
print('trainset_with_label', trainset_with_label.shape)


#
# Step 2: training a classifie ######################################################################################
def Iterative_Model_get_reliable_negative_examples(model, trainset_with_label, trainset_without_label):
    trainset_with_label_id = trainset_with_label['customer_id'].tolist()
    trainset_with_label_label = trainset_with_label['ebb_eligible'].tolist()
    del trainset_with_label['customer_id']
    del trainset_with_label['ebb_eligible']
    trainset_without_label_id = trainset_without_label['customer_id'].tolist()
    trainset_without_label_label = trainset_without_label['ebb_eligible'].tolist()
    del trainset_without_label['customer_id']
    del trainset_without_label['ebb_eligible']
    # 2.1. 用model 去predict trainset_without_label
    y = trainset_with_label_label
    X = trainset_with_label
    X_test = trainset_without_label
    print('Model fitting')
    model.fit(X, y)

    print('Model predicting')
    # 对得到trainset_without_label没有label的数据进行predict，得到结果
    y_predict = model.predict(X_test)
    # print(y_predict)
    print('number of zero:', list(y_predict).count(negative_label))
    print('number of one:', list(y_predict).count(positive_label))
    # predict_res DataFrame{customer_id, ebb_eligible}
    # 预测结果及 有0也有1
    predict_res = pd.DataFrame()
    predict_res['customer_id'] = trainset_without_label_id
    predict_res['ebb_eligible'] = y_predict
    # print('predict_res\n', predict_res)
    # 写回trainset_with_label， trainset_without_label的customer_id和ebb_eligible
    trainset_with_label['customer_id'] = trainset_with_label_id
    trainset_with_label['ebb_eligible'] = trainset_with_label_label
    trainset_without_label['customer_id'] = trainset_without_label_id
    trainset_without_label['ebb_eligible'] = trainset_without_label_label
    # 预测为0 的 reliable_negative_examples
    trainset_with_label_zero = pd.DataFrame(predict_res[predict_res['ebb_eligible'] == negative_label]['customer_id'])
    # print('label_zero_predict_res\n', trainset_with_label_zero)
    print('Shape (solid negative ))', trainset_with_label_zero.shape)
    trainset_with_label_zero = pd.merge(trainset_with_label_zero, trainset_without_label, on='customer_id')
    trainset_with_label_zero['ebb_eligible'] = negative_label
    # 2.2. 把标label=0的数据 加入trainset_with_label
    print('Shape trainset_with_label before:', trainset_with_label.shape)
    trainset_with_label = pd.concat([trainset_with_label, trainset_with_label_zero])
    print('Shape trainset_with_label after:', trainset_with_label.shape)
    # 2.3.把标label=0的数据 删除从trainset_without_label
    print('Shape trainset_without_label before:', trainset_without_label.shape)
    trainset_without_label = pd.concat([trainset_without_label, trainset_with_label_zero]).drop_duplicates(
        subset=['customer_id'], keep=False)
    print('Shape trainset_without_label after:', trainset_without_label.shape)
    return model, trainset_with_label, trainset_without_label
    # 循环以上操作，直到
    # 达到最大迭代次数
    # 本次迭代没有数据被标为label = 0


# iter次数大于10缓慢学习solid negative
def Slow_Iterative_Model_get_reliable_negative_examples(model, trainset_with_label, trainset_without_label):
    trainset_with_label_id = trainset_with_label['customer_id'].tolist()
    trainset_with_label_label = trainset_with_label['ebb_eligible'].tolist()
    del trainset_with_label['customer_id']
    del trainset_with_label['ebb_eligible']
    trainset_without_label_id = trainset_without_label['customer_id'].tolist()
    trainset_without_label_label = trainset_without_label['ebb_eligible'].tolist()
    del trainset_without_label['customer_id']
    del trainset_without_label['ebb_eligible']
    # 2.1. 用model 去predict trainset_without_label
    y = trainset_with_label_label
    X = trainset_with_label
    X_test = trainset_without_label
    print('Model fitting')
    model.fit(X, y)
    # print('model class',model.classes_)
    print('Model predicting')
    # 对得到trainset_without_label没有label的数据进行predict，得到结果
    y_predict = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print('number of zero:', list(y_predict).count(negative_label))
    print('number of one:', list(y_predict).count(positive_label))
    # predict_res DataFrame{customer_id, ebb_eligible}
    # 预测结果及 有0也有1
    predict_res = pd.DataFrame()
    predict_res['customer_id'] = trainset_without_label_id
    predict_res['ebb_eligible'] = y_predict
    predict_res[f'prob_label_{model.classes_[0]}'] = y_prob[:,
                                                     0]  # y_predict[:,0]是label=1的概率， y_predict[:,1]是label=0的概率
    predict_res[f'prob_label_{model.classes_[1]}'] = y_prob[:, 1]
    # print('predict_res\n', predict_res)
    # 写回trainset_with_label， trainset_without_label的customer_id和ebb_eligible
    trainset_with_label['customer_id'] = trainset_with_label_id
    trainset_with_label['ebb_eligible'] = trainset_with_label_label
    trainset_without_label['customer_id'] = trainset_without_label_id
    trainset_without_label['ebb_eligible'] = trainset_without_label_label
    # 预测为0 的 reliable_negative_examples
    predict_res = pd.DataFrame(predict_res[predict_res['ebb_eligible'] == negative_label])
    # 对他们按照prob排序, label=0 (-1) 的概率升序排序， prob越大，越可能为0
    if model.classes_[0] == -1:
        # print('第一个是 -1')
        predict_res = predict_res.sort_values(by=f'prob_label_{model.classes_[0]}', ascending=True)
    else:
        # print('第二个是 -1')
        predict_res = predict_res.sort_values(by=f'prob_label_{model.classes_[1]}', ascending=True)

    # 每次只截取一半， 有100个为0的，只截取50个prob最大的
    predict_res = predict_res.iloc[len(predict_res) // 2:, :]
    # 找出trainset_with_label_zero的customer_id
    trainset_with_label_zero = pd.DataFrame(predict_res['customer_id'])
    # print('label_zero_predict_res\n', trainset_with_label_zero)
    print('number of zero be chosen:', trainset_with_label_zero.shape)
    trainset_with_label_zero = pd.merge(trainset_with_label_zero, trainset_without_label, on='customer_id')
    trainset_with_label_zero['ebb_eligible'] = negative_label
    # 2.2. 把标label=0的数据 加入trainset_with_label
    print('Shape trainset_with_label before:', trainset_with_label.shape)
    trainset_with_label = pd.concat([trainset_with_label, trainset_with_label_zero])
    print('Shape trainset_with_label after:', trainset_with_label.shape)
    # 2.3.把标label=0的数据 删除从trainset_without_label
    print('Shape trainset_without_label before:', trainset_without_label.shape)
    trainset_without_label = pd.concat([trainset_without_label, trainset_with_label_zero]).drop_duplicates(
        subset=['customer_id'], keep=False)
    print('Shape trainset_without_label after:', trainset_without_label.shape)
    return model, trainset_with_label, trainset_without_label
    # 循环以上操作，直到
    # 达到最大迭代次数
    # 本次迭代没有数据被标为label = 0


max_iter = 1000
model = SVC(probability=True, degree=4)
for i in range(max_iter):
    print(f'Iterative {i}')
    len_trainset_with_label = len(trainset_with_label)
    # 前10次正常找solid
    if i <= 10:
        model, trainset_with_label, trainset_without_label = Iterative_Model_get_reliable_negative_examples(model,
                                                                                                            trainset_with_label,
                                                                                                            trainset_without_label)

    else:
        model, trainset_with_label, trainset_without_label = Slow_Iterative_Model_get_reliable_negative_examples(model,
                                                                                                                 trainset_with_label,
                                                                                                                 trainset_without_label)
    joblib.dump(model, f'./Model/{curr_time}_{i}_Model.joblib')
    print('\n')
    # 当没有新的label=0的数据被作为reliable negative examples时，停止
    # 或 early stop 当新增的少于20时，停止
    print()
    if len(trainset_with_label) - len_trainset_with_label <= 20:
        print('No new reliable negative examples')
        break

# model = joblib.load(f'./Model/{curr_time}_model.joblib')
# 剩下的trainset_without_label的label都是1
trainset_without_label['ebb_eligible'] = positive_label
# 把剩下label=1的trainset_without_label的数据加入trainset_with_label
# 现在所有数据都是有标签的
trainset_with_label = pd.concat([trainset_with_label, trainset_without_label], ignore_index=True)
# 对已经全部有标签的trainset训练
trainset_with_label_id = trainset_with_label['customer_id']
del trainset_with_label['customer_id']
y = trainset_with_label['ebb_eligible']
del trainset_with_label['ebb_eligible']
X = trainset_with_label

# 测试一下模型
print('Test Model')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('Model fitting')
model.fit(X_train, y_train)
print('Model fitting')
y_predict = model.predict(X_test)

print('classification_report\n', classification_report(y_test, y_predict))
print('confusion_matrix\n', confusion_matrix(y_test, y_predict))

# 预测evalset
print('Predict Evalset')
testset_label = testset['customer_id']
del testset['customer_id']
X_eval = testset
print('Model fitting')
model.fit(X, y)  # 用整体数据训练
print('Model predicting')
predict_y = model.predict(X_eval)
# 保存模型
joblib.dump(model, f'./Model/{curr_time}_{i + 1}_FinalSVMiterModel.joblib')
print(f'Model Saved at: ./Model/{curr_time}_{i + 1}_FinalSVMiterModel.joblib')

# 写回trainset_with_label, 只有在写回的时候negative label由-1变为0
res = pd.DataFrame()
res['customer_id'] = testset_label
res['ebb_eligible'] = [1 if x == 1 else 0 for x in predict_y]

trainset_with_label['customer_id'] = trainset_with_label_id
trainset_with_label['ebb_eligible'] = [1 if x == 1 else 0 for x in y]

# 预测结果
print('Number of one in train ', y.tolist().count(positive_label))
print('Number of zero in train ', y.tolist().count(negative_label))
print('Number of one in eval ', list(predict_y).count(positive_label))
print('Number of zero in eval ', list(predict_y).count(negative_label))
res.to_csv(f'./res/{curr_time}.csv', index=False)
# 标记过的trainset ,大小与原本trainset一样
trainset_with_label.to_csv('./Features/TrainWithLabel.csv')
print(f'Result Save at ./res/{curr_time}.csv')
