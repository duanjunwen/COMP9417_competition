# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
import time

import nni
import logging
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

LOG = logging.getLogger('sklearn_classification')
curr_time = time.strftime("%Y-%m-%d", time.localtime())
ModelName = 'Model_Cat'
ModelSavePath = f'./Model/{ModelName}'
ResSavePath = f'./res/{ModelName}'

ModelDict = os.path.exists('./Model')
if not ModelDict:
    os.makedirs('./Model')
Model_Dict = os.path.exists(ModelSavePath)
if not Model_Dict:
    os.makedirs(ModelSavePath)
# evalset结果存储位置
ResPath = os.path.exists('./res')
if not ResPath:
    os.makedirs('./res')
Model_Res_Path = os.path.exists(ResSavePath)
if not Model_Res_Path:
    os.makedirs(ResSavePath)


def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    dataset = pd.read_csv('./Features/TrainWithLabelCat.csv', index_col=[0])
    y = dataset['ebb_eligible'].tolist()
    y = [1 if value == 1 else -1 for value in y]
    del dataset['ebb_eligible']
    del dataset['customer_id']
    X = dataset
    # # balance Data
    # smote = SMOTE()
    # X, y = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


def get_default_parameters():
    '''get default parameters'''
    params = {'iterations':10000, 'verbose':1000, 'eval_metric':'F1', 'depth':7, 'l2_leaf_reg':7, 'learning_rate':0.07675896435626739}
    return params


def get_model(PARAMS):
    '''Get model according to parameters'''
    model = CatBoostClassifier(iterations=5000, verbose=1000, eval_metric='F1', bootstrap_type='Bernoulli')
    params = {'depth': PARAMS.get('depth'), 'l2_leaf_reg': PARAMS.get('l2_leaf_reg'),
              'learning_rate': PARAMS.get('learning_rate'), 'random_strength':PARAMS.get('random_strength')}
    model.set_params(**params)
    return model


def run(X_train, X_test, y_train, y_test, model):
    '''Train model and predict result'''
    print('Model fit..')
    model.fit(X_train, y_train)
    print('Model predict...')
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average=None)
    confuse_matrix = confusion_matrix(y_test, y_pred)
    print('Classification report\n', classification_report(y_test, y_pred))
    print('confusion_matrix\n', confuse_matrix)
    print('Wrong case in 0 is', confuse_matrix[0,1])
    print('Wrong case in 1 is ', confuse_matrix[1,0])

    # 预测evalset
    evalset = pd.read_csv('./Features/EvalFeatureSelected.csv', index_col=[0])
    eval_customer_id = evalset['customer_id'].tolist()
    del evalset['customer_id']
    print('Predict Eval Set:')
    eval_perdict = model.predict(evalset)

    print('Number of one in eval ', list(eval_perdict).count(1))
    print('Number of zero in eval ', list(eval_perdict).count(-1))

    res = pd.DataFrame()
    res['customer_id'] = eval_customer_id
    res['ebb_eligible'] = [1 if value == 1 else 0 for value in eval_perdict]
    res.to_csv(f'{ResSavePath}/{curr_time}_{ModelName}.csv', index=False)


    LOG.debug('score: %s', score)
    LOG.debug('Total wrong case: %s', confuse_matrix[0,1] + confuse_matrix[1,0])
    nni.report_final_result(confuse_matrix[0,1] + confuse_matrix[1,0])


if __name__ == '__main__':
    RECEIVED_PARAMS = nni.get_next_parameter()
    LOG.debug(RECEIVED_PARAMS)
    PARAMS = get_default_parameters()
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
