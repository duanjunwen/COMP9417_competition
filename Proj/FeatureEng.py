import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def writeFeatureMap(path, featureMap):
    featureMap_json = json.dumps(featureMap)
    with open(path, "w+") as file:
        file.write(featureMap_json)


# Feature Eng of activations_ebb_set.csv
def ActivationFeature():
    data1 = pd.read_csv('./data/activations_ebb_set1.csv')
    data2 = pd.read_csv('./data/activations_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['activation_date']

    # FeatureMap存放当前表所有特征的map关系
    FeatureMap = dict()
    featuresMapPath = './FeatureMap/ActivationFeatureMap.json'

    # activation_channel
    activation_channel_features = list(set(data['activation_channel'].tolist()))
    activation_channel_featuresMap = {activation_channel_features[i]: i for i in
                                      range(len(activation_channel_features))}
    # print('featuresMap', activation_channel_featuresMap)

    NewData = data.groupby('customer_id')['activation_channel'].apply(
        lambda x: (x == activation_channel_features[0]).sum()).reset_index(
        name=f'{activation_channel_features[0]}')
    for i in range(1, len(activation_channel_features)):
        CurrData = data.groupby('customer_id')['activation_channel'].apply(
            lambda x: (x == activation_channel_features[i]).sum()).reset_index(name=f'{activation_channel_features[i]}')
        NewData = pd.merge(NewData, CurrData, on='customer_id')

    # 保存每个特征的map
    FeatureMap['activation_channel'] = activation_channel_featuresMap
    # 把FeatureMap写入对应位置json
    writeFeatureMap(featuresMapPath, FeatureMap)

    # 接触客服的次数
    activation_times = data.groupby('customer_id').size().reset_index(name='activation_times')
    NewData = pd.merge(NewData, activation_times, on='customer_id')

    print('ActivationFeature\n', NewData.head())
    print('Shape\n', NewData.shape)

    return NewData


# ActivationFeature()

def ActivationFeature_eval():
    data = pd.read_csv('./data/activations_eval_set.csv')
    del data['activation_date']
    ActivationFeatureMap = dict()
    with open('./FeatureMap/ActivationFeatureMap.json') as file:
        ActivationFeatureMap = json.load(file)

    # activation_channel
    activation_channel_features_json = ActivationFeatureMap['activation_channel']
    activation_channel_features_list = list(activation_channel_features_json.keys())
    # print(list(activation_channel_features_json.keys()))
    NewData = data.groupby('customer_id')['activation_channel'].apply(
        lambda x: (x == activation_channel_features_list[0]).sum()).reset_index(
        name=f'{activation_channel_features_list[0]}')
    for i in range(1, len(activation_channel_features_list)):
        CurrData = data.groupby('customer_id')['activation_channel'].apply(
            lambda x: (x == activation_channel_features_list[i]).sum()).reset_index(
            name=f'{activation_channel_features_list[i]}')
        NewData = pd.merge(NewData, CurrData, on='customer_id')
    # 接触客服的次数
    activation_times = data.groupby('customer_id').size().reset_index(name='activation_times')
    NewData = pd.merge(NewData, activation_times, on='customer_id')
    print('ActivationFeature_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    # 13个feature，但是列不一样
    return NewData


# ActivationFeature_eval()


# AutoRefill
# Abandon
def AutoRefillFeatures():
    return None


def AutoRefillFeatures_eval():
    return None


# deactivations_ebb_set1.csv
def DeactivationsFeature():
    data1 = pd.read_csv('./data/deactivations_ebb_set1.csv')
    data2 = pd.read_csv('./data/deactivations_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['deactivation_date']

    # FeatureMap存放当前表所有特征的map关系
    FeatureMap = dict()
    featuresMapPath = './FeatureMap/DeactivationsFeature.json'

    # deactivation_reason
    deactivation_reason_features = list(set(data['deactivation_reason'].tolist()))
    deactivation_reason_featuresMap = {deactivation_reason_features[i]: i for i in
                                       range(len(deactivation_reason_features))}
    # print(deactivation_reason_featuresMap)
    NewData = data.groupby('customer_id')['deactivation_reason'].apply(
        lambda x: (x == deactivation_reason_features[0]).sum()).reset_index(
        name=f'{deactivation_reason_features[0]}')
    for i in range(1, len(deactivation_reason_features)):
        CurrData = data.groupby('customer_id')['deactivation_reason'].apply(
            lambda x: (x == deactivation_reason_features[i]).sum()).reset_index(
            name=f'{deactivation_reason_features[i]}')
        NewData = pd.merge(NewData, CurrData, on='customer_id')
    print('DeactivationsFeature\n', NewData.head())
    print('Shape\n', NewData.shape)

    # 保存每个特征的map
    FeatureMap['deactivation_reason'] = deactivation_reason_featuresMap
    # 把FeatureMap写入对应位置json
    writeFeatureMap(featuresMapPath, FeatureMap)

    return NewData


# DeactivationsFeature()


def DeactivationsFeature_eval():
    data = pd.read_csv('./data/deactivations_eval_set.csv')
    del data['deactivation_date']

    DeactivationsFeatureMap = dict()
    with open('./FeatureMap/DeactivationsFeature.json') as file:
        DeactivationsFeatureMap = json.load(file)
    deactivation_reason_features_json = DeactivationsFeatureMap['deactivation_reason']
    deactivation_reason_features_json_list = list(deactivation_reason_features_json.keys())
    NewData = data.groupby('customer_id')['deactivation_reason'].apply(
        lambda x: (x == deactivation_reason_features_json_list[0]).sum()).reset_index(
        name=f'{deactivation_reason_features_json_list[0]}')
    for i in range(1, len(deactivation_reason_features_json_list)):
        CurrData = data.groupby('customer_id')['deactivation_reason'].apply(
            lambda x: (x == deactivation_reason_features_json_list[i]).sum()).reset_index(
            name=f'{deactivation_reason_features_json_list[i]}')
        NewData = pd.merge(NewData, CurrData, on='customer_id')
    print('DeactivationsFeature_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    # 27个feature
    return NewData


# DeactivationsFeature_eval()

# deprioritizing_ebb_set1.csv
def DeprioritizingFeatures():
    return None


def DeprioritizingFeatures_eval():
    return None


# InteractionsFeature()
# 涉及文本处理
def InteractionsFeature():
    data1 = pd.read_csv('./data/interactions_ebb_set1.csv')
    data2 = pd.read_csv('./data/interactions_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['date']

    # FeatureMap存放当前表所有特征的map关系
    FeatureMap = dict()
    featuresMapPath = './FeatureMap/InteractionsFeature.json'

    # issue
    # IssueFeatures = list(set(data['issue'].tolist()))
    # IssueFeaturesMap = {IssueFeatures[i]: i for i in range(len(IssueFeatures))}

    # reason
    # ReasonFeatures = list(set(data['reason'].tolist()))
    # ReasonFeaturesMap = {ReasonFeatures[i]: i for i in range(len(ReasonFeatures))}

    # disposition
    # 1.data['disposition'] lower case and change non to None
    data['disposition'] = data['disposition'].str.lower()
    data['disposition'] = data['disposition'].where(pd.notnull(data['disposition']), None)
    disposition_features = list(set(data['disposition'].tolist()))

    # 2.Transfor to {'schedule callback': 0, 'call terminated': 1, None: 2, 'transferred': 3, 'successful': 4}
    disposition_NameMap = dict()
    for featurename in disposition_features:
        featurename = str(featurename)
        # print('featurename:', featurename)
        if featurename is not None:
            # unsuccessful schedule callback 和 call terminated， transferred是同一类
            if 'unsuccessful' in featurename:
                disposition_NameMap[featurename] = 'unsuccessful'
            elif 'successful' in featurename:
                disposition_NameMap[featurename] = 'successful'
            elif 'suc' in featurename:
                disposition_NameMap[featurename] = 'successful'
            elif 'ticket created' in featurename:
                disposition_NameMap[featurename] = 'successful'

            elif 'schedule callback' in featurename:
                disposition_NameMap[featurename] = 'schedule callback'
            elif 'call terminated' in featurename:
                disposition_NameMap[featurename] = 'call terminated'
            elif 'transferred' in featurename:
                disposition_NameMap[featurename] = 'transferred'
            else:
                disposition_NameMap[featurename] = None
        else:
            disposition_NameMap[featurename] = None
    # print('disposition_NameMap:', disposition_NameMap)
    # 把原本不规则文本map成处理后的文本
    data["disposition"].replace(disposition_NameMap, inplace=True)
    New_disposition_features = list(set(data['disposition'].tolist()))
    # 把处理后的文本map成int，需要保存在FeatureMap中
    disposition_featuresMap = {New_disposition_features[i]: i for i in
                               range(len(New_disposition_features))}
    reverse_disposition_featuresMap = {value: key for key, value in disposition_featuresMap.items()}
    # print('disposition_featuresMap:', disposition_featuresMap)
    # print('reverse_disposition_featuresMap:', reverse_disposition_featuresMap)

    data["disposition"].replace(disposition_featuresMap, inplace=True)
    # ["disposition"]只有6种情况，分情况统计
    # {0: 'transferred', 1: 'schedule callback', 2: 'successful', 3: None, 4: 'unsuccessful', 5: 'call terminated'}

    # Feature 0
    Feature_0 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 0).sum()).reset_index(
        name=reverse_disposition_featuresMap[0])
    NewData = Feature_0
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)
    # Feature 1
    Feature_1 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 1).sum()).reset_index(
        name=reverse_disposition_featuresMap[1])
    NewData = pd.merge(NewData, Feature_1, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 2
    Feature_2 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 2).sum()).reset_index(
        name=reverse_disposition_featuresMap[2])
    NewData = pd.merge(NewData, Feature_2, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 3
    Feature_3 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 3).sum()).reset_index(
        name=reverse_disposition_featuresMap[3])
    NewData = pd.merge(NewData, Feature_3, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 4
    Feature_4 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 4).sum()).reset_index(
        name=reverse_disposition_featuresMap[4])
    NewData = pd.merge(NewData, Feature_4, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 5
    Feature_5 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 5).sum()).reset_index(
        name=reverse_disposition_featuresMap[5])
    NewData = pd.merge(NewData, Feature_5, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    print('InteractionsFeature\n', NewData.head())
    print('Shape\n', NewData.shape)

    # 保存每个特征的map
    FeatureMap['disposition'] = disposition_featuresMap
    # 把FeatureMap写入对应位置json
    writeFeatureMap(featuresMapPath, FeatureMap)

    return NewData


# InteractionsFeature()

def InteractionsFeature_eval():
    data = pd.read_csv('./data/interactions_eval_set.csv')
    del data['date']

    InteractionsFeatureMap = dict()
    with open('./FeatureMap/InteractionsFeature.json') as file:
        InteractionsFeatureMap = json.load(file)
    # print('InteractionsFeatureMap', InteractionsFeatureMap)
    dispositionFeatureMap = InteractionsFeatureMap['disposition']
    if 'null' in dispositionFeatureMap.keys():
        value = dispositionFeatureMap['null']
        del dispositionFeatureMap['null']
        dispositionFeatureMap[None] = value
    # print('dispositionFeatureMap', dispositionFeatureMap)
    # disposition
    # 1.data['disposition'] lower case and change non to None
    data['disposition'] = data['disposition'].str.lower()
    data['disposition'] = data['disposition'].where(pd.notnull(data['disposition']), None)
    disposition_features = list(set(data['disposition'].tolist()))
    # 2.Transfor to {'schedule callback': 0, 'call terminated': 1, None: 2, 'transferred': 3, 'successful': 4}
    disposition_NameMap = dict()
    for featurename in disposition_features:
        featurename = str(featurename)
        # print('featurename:', featurename)
        if featurename is not None:
            if 'unsuccessful' in featurename:
                disposition_NameMap[featurename] = 'unsuccessful'
            elif 'successful' in featurename:
                disposition_NameMap[featurename] = 'successful'
            elif 'suc' in featurename:
                disposition_NameMap[featurename] = 'successful'
            elif 'ticket created' in featurename:
                disposition_NameMap[featurename] = 'successful'
            elif 'schedule callback' in featurename:
                disposition_NameMap[featurename] = 'schedule callback'
            elif 'call terminated' in featurename:
                disposition_NameMap[featurename] = 'call terminated'
            elif 'transferred' in featurename:
                disposition_NameMap[featurename] = 'transferred'
            else:
                disposition_NameMap[featurename] = None
        else:
            disposition_NameMap[featurename] = None
    # print('disposition_NameMap:', disposition_NameMap)
    # 把原本不规则文本map成处理后的文本
    data["disposition"].replace(disposition_NameMap, inplace=True)
    # 把处理后的文本map成int，需要保存在FeatureMap中
    disposition_featuresMap = dispositionFeatureMap
    # print('disposition_featuresMap', disposition_featuresMap)
    reverse_disposition_featuresMap = {value: key for key, value in disposition_featuresMap.items()}
    data["disposition"].replace(disposition_featuresMap, inplace=True)
    # print(list(set(data['disposition'].tolist())))
    # ["disposition"]只有6种情况，分情况统计
    # {0: 'transferred', 1: 'schedule callback', 2: 'successful', 3: None, 4: 'unsuccessful', 5: 'call terminated'}

    # Feature 0
    Feature_0 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 0).sum()).reset_index(
        name=reverse_disposition_featuresMap[0])
    NewData = Feature_0
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)
    # Feature 1
    Feature_1 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 1).sum()).reset_index(
        name=reverse_disposition_featuresMap[1])
    NewData = pd.merge(NewData, Feature_1, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 2
    Feature_2 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 2).sum()).reset_index(
        name=reverse_disposition_featuresMap[2])
    NewData = pd.merge(NewData, Feature_2, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 3
    Feature_3 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 3).sum()).reset_index(
        name=reverse_disposition_featuresMap[3])
    NewData = pd.merge(NewData, Feature_3, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 4
    Feature_4 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 4).sum()).reset_index(
        name=reverse_disposition_featuresMap[4])
    NewData = pd.merge(NewData, Feature_4, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    # Feature 5
    Feature_5 = data.groupby('customer_id')['disposition'].apply(lambda x: (x == 5).sum()).reset_index(
        name=reverse_disposition_featuresMap[5])
    NewData = pd.merge(NewData, Feature_5, on='customer_id')
    # print('NewData\n', NewData.head())
    # print('Shanpe', NewData.shape)

    print('InteractionsFeature_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# InteractionsFeature_eval()


# ivr_calls_ebb_set1.csv
# InteractionsFeature()
def IVRCallsFeatures():
    data1 = pd.read_csv('./data/ivr_calls_ebb_set1.csv')
    data2 = pd.read_csv('./data/ivr_calls_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['call_start_time']
    del data['call_end_time']
    # print(data.head())
    ## iscompleted有三种情况 NotCompleted, Completed, IsNone
    data1 = data.groupby('customer_id')['iscompleted'].apply(lambda x: (x == 0).sum()).reset_index(name='NotCompleted')
    data2 = data.groupby('customer_id')['iscompleted'].apply(lambda x: (x == 1).sum()).reset_index(name='Completed')
    # isNone可以舍弃
    data3 = data.groupby('customer_id')['iscompleted'].apply(lambda x: (x == None).sum()).reset_index(name='IsNone')
    NewData = pd.merge(pd.merge(data1, data2, on='customer_id'), data3, on='customer_id')
    print('IVRCallsFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


def IVRCallsFeatures_eval():
    data = pd.read_csv('./data/ivr_calls_eval_set.csv')
    del data['call_start_time']
    del data['call_end_time']
    # print(data.head())
    ## iscompleted有三种情况 NotCompleted, Completed, IsNone
    data1 = data.groupby('customer_id')['iscompleted'].apply(lambda x: (x == 0).sum()).reset_index(name='NotCompleted')
    data2 = data.groupby('customer_id')['iscompleted'].apply(lambda x: (x == 1).sum()).reset_index(name='Completed')
    # isNone可以舍弃
    data3 = data.groupby('customer_id')['iscompleted'].apply(lambda x: (x == None).sum()).reset_index(name='IsNone')
    NewData = pd.merge(pd.merge(data1, data2, on='customer_id'), data3, on='customer_id')
    print('IVRCallsFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# lease_history_ebb_set1.csv
def LeaseHistoryFeatures():
    return None


def LeaseHistoryFeatures_eval():
    return None


def LoyaltyprogramFeatures():
    data1 = pd.read_csv('./data/loyalty_program_ebb_set1.csv')
    data2 = pd.read_csv('./data/loyalty_program_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['lrp_enrolled']
    del data['date']
    NewData = data.groupby('customer_id')['total_quantity'].sum().reset_index(name='total_quantity')
    print('LoyaltyprogramFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# LoyaltyprogramFeatures()

def LoyaltyprogramFeatures_eval():
    data = pd.read_csv('./data/loyalty_program_eval_set.csv')
    del data['lrp_enrolled']
    del data['date']
    NewData = data.groupby('customer_id')['total_quantity'].sum().reset_index(name='total_quantity')
    print('LoyaltyprogramFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


def NetworkFeatures():
    data1 = pd.read_csv('./data/network_ebb_set1.csv')
    data2 = pd.read_csv('./data/network_ebb_set1.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['date']
    # voice_minutes
    dataVoice_minutes = data.groupby('customer_id')['voice_minutes'].sum().reset_index(name='voice_minutes')
    # total_sms
    dataTotal_sms = data.groupby('customer_id')['total_sms'].sum().reset_index(name='total_sms')
    # total_kb
    dataTotal_kb = data.groupby('customer_id')['total_kb'].sum().reset_index(name='total_kb')
    # hotspot_kb
    dataHotspot_kb = data.groupby('customer_id')['hotspot_kb'].sum().reset_index(name='hotspot_kb')

    # print(dataVoice_minutes.head())
    # print(len(dataVoice_minutes))
    # print(dataTotal_sms.head())
    # print(len(dataTotal_sms))
    # print(dataTotal_kb.head())
    # print(len(dataTotal_kb))
    # print(dataHotspot_kb.head())
    # print(len(dataHotspot_kb))

    NewData = pd.merge(
        pd.merge(pd.merge(dataVoice_minutes, dataTotal_sms, on='customer_id'), dataTotal_kb, on='customer_id'),
        dataHotspot_kb, on='customer_id')
    print('NetworkFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


def NetworkFeatures_eval():
    data = pd.read_csv('./data/network_eval_set.csv')
    del data['date']
    # voice_minutes
    dataVoice_minutes = data.groupby('customer_id')['voice_minutes'].sum().reset_index(name='voice_minutes')
    # total_sms
    dataTotal_sms = data.groupby('customer_id')['total_sms'].sum().reset_index(name='total_sms')
    # total_kb
    dataTotal_kb = data.groupby('customer_id')['total_kb'].sum().reset_index(name='total_kb')
    # hotspot_kb
    dataHotspot_kb = data.groupby('customer_id')['hotspot_kb'].sum().reset_index(name='hotspot_kb')

    # print(dataVoice_minutes.head())
    # print(len(dataVoice_minutes))
    # print(dataTotal_sms.head())
    # print(len(dataTotal_sms))
    # print(dataTotal_kb.head())
    # print(len(dataTotal_kb))
    # print(dataHotspot_kb.head())
    # print(len(dataHotspot_kb))

    NewData = pd.merge(
        pd.merge(pd.merge(dataVoice_minutes, dataTotal_sms, on='customer_id'), dataTotal_kb, on='customer_id'),
        dataHotspot_kb, on='customer_id')
    print('NetworkFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# NetworkFeatures()

# notifying_ebb_set1.csv
def NotifyingFeatures():
    return None


def NotifyingFeatures_eval():
    return None


def PhoneDataFeatures():
    data1 = pd.read_csv('./data/phone_data_ebb_set1.csv')
    data2 = pd.read_csv('./data/phone_data_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['language']
    del data['timestamp']
    del data['time_since_last_boot']

    # # 这里不再求sum，因为sum没意义
    # battery_available
    battery_available_Avg = data.groupby('customer_id')['battery_available'].mean().reset_index(
        name='battery_available_Avg')
    battery_available_Min = data.groupby('customer_id')['battery_available'].min().reset_index(
        name='battery_available_Min')
    battery_available_Max = data.groupby('customer_id')['battery_available'].max().reset_index(
        name='battery_available_Max')

    # print(battery_available_Avg.head())
    # print(battery_available_Min.head())
    # print(battery_available_Max.head())
    # print(battery_available_Avg.shape)
    # print(battery_available_Min.shape)
    # print(battery_available_Max.shape)

    NewData = battery_available_Avg
    NewData = pd.merge(NewData, battery_available_Min, on='customer_id')
    NewData = pd.merge(NewData, battery_available_Max, on='customer_id')

    # battery_total
    battery_total_Avg = data.groupby('customer_id')['battery_total'].mean().reset_index(name='battery_total_Avg')
    battery_total_Min = data.groupby('customer_id')['battery_total'].min().reset_index(name='battery_total_Min')
    battery_total_Max = data.groupby('customer_id')['battery_total'].max().reset_index(name='battery_total_Max')

    # print(battery_total_Avg.head())
    # print(battery_total_Min.head())
    # print(battery_total_Max.head())
    # print(battery_total_Avg.shape)
    # print(battery_total_Min.shape)
    # print(battery_total_Max.shape)

    NewData = pd.merge(NewData, battery_total_Avg, on='customer_id')
    NewData = pd.merge(NewData, battery_total_Min, on='customer_id')
    NewData = pd.merge(NewData, battery_total_Max, on='customer_id')

    # # bluetooth_on
    bluetooth_on = data.groupby('customer_id')['bluetooth_on'].apply(lambda x: (x == True).sum()).reset_index(
        name='bluetooth_on')
    bluetooth_off = data.groupby('customer_id')['bluetooth_on'].apply(lambda x: (x == False).sum()).reset_index(
        name='bluetooth_off')

    # print(bluetooth_on.head())
    # print(bluetooth_off.head())
    # print(bluetooth_on.shape)
    # print(bluetooth_off.shape)

    NewData = pd.merge(NewData, bluetooth_on, on='customer_id')
    NewData = pd.merge(NewData, bluetooth_off, on='customer_id')

    # #data_roaming
    data_roaming_true = data.groupby('customer_id')['data_roaming'].apply(lambda x: (x == True).sum()).reset_index(
        name='data_roaming_true')
    data_roaming_false = data.groupby('customer_id')['data_roaming'].apply(lambda x: (x == False).sum()).reset_index(
        name='data_roaming_false')
    # print(data_roaming_true.head())
    # print(data_roaming_false.head())
    # print(data_roaming_true.shape)
    # print(data_roaming_false.shape)

    NewData = pd.merge(NewData, data_roaming_true, on='customer_id')
    NewData = pd.merge(NewData, data_roaming_false, on='customer_id')

    # memory_available
    memory_available_Avg = data.groupby('customer_id')['memory_available'].mean().reset_index(
        name='memory_available_Avg')
    memory_available_Min = data.groupby('customer_id')['memory_available'].min().reset_index(
        name='memory_available_Min')
    memory_available_Max = data.groupby('customer_id')['memory_available'].max().reset_index(
        name='memory_available_Max')

    # print(memory_available_Avg.head())
    # print(memory_available_Min.head())
    # print(memory_available_Max.head())
    # print(memory_available_Avg.shape)
    # print(memory_available_Min.shape)
    # print(memory_available_Max.shape)

    NewData = pd.merge(NewData, memory_available_Avg, on='customer_id')
    NewData = pd.merge(NewData, memory_available_Min, on='customer_id')
    NewData = pd.merge(NewData, memory_available_Max, on='customer_id')

    # # memory_total
    memory_total_Avg = data.groupby('customer_id')['memory_total'].mean().reset_index(
        name='memory_total_Avg')
    memory_total_Min = data.groupby('customer_id')['memory_total'].min().reset_index(
        name='memory_total_Min')
    memory_total_Max = data.groupby('customer_id')['memory_total'].max().reset_index(
        name='memory_total_Max')

    # print(memory_total_Avg.head())
    # print(memory_total_Min.head())
    # print(memory_total_Max.head())
    # print(memory_total_Avg.shape)
    # print(memory_total_Min.shape)
    # print(memory_total_Max.shape)

    NewData = pd.merge(NewData, memory_total_Avg, on='customer_id')
    NewData = pd.merge(NewData, memory_total_Min, on='customer_id')
    NewData = pd.merge(NewData, memory_total_Max, on='customer_id')

    # # memory_treshold
    memory_treshold_Avg = data.groupby('customer_id')['memory_treshold'].mean().reset_index(
        name='memory_treshold_Avg')
    memory_treshold_Min = data.groupby('customer_id')['memory_treshold'].min().reset_index(
        name='memory_treshold_Min')
    memory_treshold_Max = data.groupby('customer_id')['memory_treshold'].max().reset_index(
        name='memory_treshold_Max')

    # print(memory_treshold_Avg.head())
    # print(memory_treshold_Min.head())
    # print(memory_treshold_Max.head())
    # print(memory_treshold_Avg.shape)
    # print(memory_treshold_Min.shape)
    # print(memory_treshold_Max.shape)

    NewData = pd.merge(NewData, memory_treshold_Avg, on='customer_id')
    NewData = pd.merge(NewData, memory_treshold_Min, on='customer_id')
    NewData = pd.merge(NewData, memory_treshold_Max, on='customer_id')

    # 不使用
    # # #sd_storage_present
    # sd_storage_present_Yes = data.groupby('customer_id')['sd_storage_present'].apply(lambda x: (x == True).sum()).reset_index(name='sd_storage_present_Yes')
    # sd_storage_present_No = data.groupby('customer_id')['sd_storage_present'].apply(lambda x: (x == False).sum()).reset_index(name='sd_storage_present_No')
    # print(sd_storage_present_Yes.head())
    # print(sd_storage_present_No.head())

    # # storage_available
    storage_available_Avg = data.groupby('customer_id')['storage_available'].mean().reset_index(
        name='storage_available_Avg')
    storage_available_Min = data.groupby('customer_id')['storage_available'].min().reset_index(
        name='storage_available_Min')
    storage_available_Max = data.groupby('customer_id')['storage_available'].max().reset_index(
        name='storage_available_Max')

    # print(storage_available_Avg.head())
    # print(storage_available_Min.head())
    # print(storage_available_Max.head())
    # print(storage_available_Avg.shape)
    # print(storage_available_Min.shape)
    # print(storage_available_Max.shape)

    NewData = pd.merge(NewData, storage_available_Avg, on='customer_id')
    NewData = pd.merge(NewData, storage_available_Min, on='customer_id')
    NewData = pd.merge(NewData, storage_available_Max, on='customer_id')

    # # storage_total
    storage_total_Avg = data.groupby('customer_id')['storage_total'].mean().reset_index(
        name='storage_total_Avg')
    storage_total_Min = data.groupby('customer_id')['storage_total'].min().reset_index(
        name='storage_total_Min')
    storage_total_Max = data.groupby('customer_id')['storage_total'].max().reset_index(
        name='storage_total_Max')

    # print(storage_total_Avg.head())
    # print(storage_total_Min.head())
    # print(storage_total_Max.head())
    # print(storage_total_Avg.shape)
    # print(storage_total_Min.shape)
    # print(storage_total_Max.shape)

    NewData = pd.merge(NewData, storage_total_Avg, on='customer_id')
    NewData = pd.merge(NewData, storage_total_Min, on='customer_id')
    NewData = pd.merge(NewData, storage_total_Max, on='customer_id')

    # # temperature
    temperature_Avg = data.groupby('customer_id')['temperature'].mean().reset_index(
        name='temperature_Avg')
    temperature_Min = data.groupby('customer_id')['temperature'].min().reset_index(
        name='temperature_Min')
    temperature_Max = data.groupby('customer_id')['temperature'].max().reset_index(
        name='temperature_Max')

    # print(temperature_Avg.head())
    # print(temperature_Min.head())
    # print(temperature_Max.head())
    # print(temperature_Avg.shape)
    # print(temperature_Min.shape)
    # print(temperature_Max.shape)

    NewData = pd.merge(NewData, temperature_Avg, on='customer_id')
    NewData = pd.merge(NewData, temperature_Min, on='customer_id')
    NewData = pd.merge(NewData, temperature_Max, on='customer_id')

    print('PhoneDataFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)

    return NewData


def PhoneDataFeatures_eval():
    data = pd.read_csv('./data/phone_data_eval_set.csv')
    del data['language']
    del data['timestamp']
    del data['time_since_last_boot']

    # # 这里不再求sum，因为sum没意义
    # battery_available
    battery_available_Avg = data.groupby('customer_id')['battery_available'].mean().reset_index(
        name='battery_available_Avg')
    battery_available_Min = data.groupby('customer_id')['battery_available'].min().reset_index(
        name='battery_available_Min')
    battery_available_Max = data.groupby('customer_id')['battery_available'].max().reset_index(
        name='battery_available_Max')

    # print(battery_available_Avg.head())
    # print(battery_available_Min.head())
    # print(battery_available_Max.head())
    # print(battery_available_Avg.shape)
    # print(battery_available_Min.shape)
    # print(battery_available_Max.shape)

    NewData = battery_available_Avg
    NewData = pd.merge(NewData, battery_available_Min, on='customer_id')
    NewData = pd.merge(NewData, battery_available_Max, on='customer_id')

    # battery_total
    battery_total_Avg = data.groupby('customer_id')['battery_total'].mean().reset_index(name='battery_total_Avg')
    battery_total_Min = data.groupby('customer_id')['battery_total'].min().reset_index(name='battery_total_Min')
    battery_total_Max = data.groupby('customer_id')['battery_total'].max().reset_index(name='battery_total_Max')

    # print(battery_total_Avg.head())
    # print(battery_total_Min.head())
    # print(battery_total_Max.head())
    # print(battery_total_Avg.shape)
    # print(battery_total_Min.shape)
    # print(battery_total_Max.shape)

    NewData = pd.merge(NewData, battery_total_Avg, on='customer_id')
    NewData = pd.merge(NewData, battery_total_Min, on='customer_id')
    NewData = pd.merge(NewData, battery_total_Max, on='customer_id')

    # # bluetooth_on
    bluetooth_on = data.groupby('customer_id')['bluetooth_on'].apply(lambda x: (x == True).sum()).reset_index(
        name='bluetooth_on')
    bluetooth_off = data.groupby('customer_id')['bluetooth_on'].apply(lambda x: (x == False).sum()).reset_index(
        name='bluetooth_off')

    # print(bluetooth_on.head())
    # print(bluetooth_off.head())
    # print(bluetooth_on.shape)
    # print(bluetooth_off.shape)

    NewData = pd.merge(NewData, bluetooth_on, on='customer_id')
    NewData = pd.merge(NewData, bluetooth_off, on='customer_id')

    # #data_roaming
    data_roaming_true = data.groupby('customer_id')['data_roaming'].apply(lambda x: (x == True).sum()).reset_index(
        name='data_roaming_true')
    data_roaming_false = data.groupby('customer_id')['data_roaming'].apply(lambda x: (x == False).sum()).reset_index(
        name='data_roaming_false')
    # print(data_roaming_true.head())
    # print(data_roaming_false.head())
    # print(data_roaming_true.shape)
    # print(data_roaming_false.shape)

    NewData = pd.merge(NewData, data_roaming_true, on='customer_id')
    NewData = pd.merge(NewData, data_roaming_false, on='customer_id')

    # memory_available
    memory_available_Avg = data.groupby('customer_id')['memory_available'].mean().reset_index(
        name='memory_available_Avg')
    memory_available_Min = data.groupby('customer_id')['memory_available'].min().reset_index(
        name='memory_available_Min')
    memory_available_Max = data.groupby('customer_id')['memory_available'].max().reset_index(
        name='memory_available_Max')

    # print(memory_available_Avg.head())
    # print(memory_available_Min.head())
    # print(memory_available_Max.head())
    # print(memory_available_Avg.shape)
    # print(memory_available_Min.shape)
    # print(memory_available_Max.shape)

    NewData = pd.merge(NewData, memory_available_Avg, on='customer_id')
    NewData = pd.merge(NewData, memory_available_Min, on='customer_id')
    NewData = pd.merge(NewData, memory_available_Max, on='customer_id')

    # # memory_total
    memory_total_Avg = data.groupby('customer_id')['memory_total'].mean().reset_index(
        name='memory_total_Avg')
    memory_total_Min = data.groupby('customer_id')['memory_total'].min().reset_index(
        name='memory_total_Min')
    memory_total_Max = data.groupby('customer_id')['memory_total'].max().reset_index(
        name='memory_total_Max')

    # print(memory_total_Avg.head())
    # print(memory_total_Min.head())
    # print(memory_total_Max.head())
    # print(memory_total_Avg.shape)
    # print(memory_total_Min.shape)
    # print(memory_total_Max.shape)

    NewData = pd.merge(NewData, memory_total_Avg, on='customer_id')
    NewData = pd.merge(NewData, memory_total_Min, on='customer_id')
    NewData = pd.merge(NewData, memory_total_Max, on='customer_id')

    # # memory_treshold
    memory_treshold_Avg = data.groupby('customer_id')['memory_treshold'].mean().reset_index(
        name='memory_treshold_Avg')
    memory_treshold_Min = data.groupby('customer_id')['memory_treshold'].min().reset_index(
        name='memory_treshold_Min')
    memory_treshold_Max = data.groupby('customer_id')['memory_treshold'].max().reset_index(
        name='memory_treshold_Max')

    # print(memory_treshold_Avg.head())
    # print(memory_treshold_Min.head())
    # print(memory_treshold_Max.head())
    # print(memory_treshold_Avg.shape)
    # print(memory_treshold_Min.shape)
    # print(memory_treshold_Max.shape)

    NewData = pd.merge(NewData, memory_treshold_Avg, on='customer_id')
    NewData = pd.merge(NewData, memory_treshold_Min, on='customer_id')
    NewData = pd.merge(NewData, memory_treshold_Max, on='customer_id')

    # 不使用
    # # #sd_storage_present
    # sd_storage_present_Yes = data.groupby('customer_id')['sd_storage_present'].apply(lambda x: (x == True).sum()).reset_index(name='sd_storage_present_Yes')
    # sd_storage_present_No = data.groupby('customer_id')['sd_storage_present'].apply(lambda x: (x == False).sum()).reset_index(name='sd_storage_present_No')
    # print(sd_storage_present_Yes.head())
    # print(sd_storage_present_No.head())

    # # storage_available
    storage_available_Avg = data.groupby('customer_id')['storage_available'].mean().reset_index(
        name='storage_available_Avg')
    storage_available_Min = data.groupby('customer_id')['storage_available'].min().reset_index(
        name='storage_available_Min')
    storage_available_Max = data.groupby('customer_id')['storage_available'].max().reset_index(
        name='storage_available_Max')

    # print(storage_available_Avg.head())
    # print(storage_available_Min.head())
    # print(storage_available_Max.head())
    # print(storage_available_Avg.shape)
    # print(storage_available_Min.shape)
    # print(storage_available_Max.shape)

    NewData = pd.merge(NewData, storage_available_Avg, on='customer_id')
    NewData = pd.merge(NewData, storage_available_Min, on='customer_id')
    NewData = pd.merge(NewData, storage_available_Max, on='customer_id')

    # # storage_total
    storage_total_Avg = data.groupby('customer_id')['storage_total'].mean().reset_index(
        name='storage_total_Avg')
    storage_total_Min = data.groupby('customer_id')['storage_total'].min().reset_index(
        name='storage_total_Min')
    storage_total_Max = data.groupby('customer_id')['storage_total'].max().reset_index(
        name='storage_total_Max')

    # print(storage_total_Avg.head())
    # print(storage_total_Min.head())
    # print(storage_total_Max.head())
    # print(storage_total_Avg.shape)
    # print(storage_total_Min.shape)
    # print(storage_total_Max.shape)

    NewData = pd.merge(NewData, storage_total_Avg, on='customer_id')
    NewData = pd.merge(NewData, storage_total_Min, on='customer_id')
    NewData = pd.merge(NewData, storage_total_Max, on='customer_id')

    # # temperature
    temperature_Avg = data.groupby('customer_id')['temperature'].mean().reset_index(
        name='temperature_Avg')
    temperature_Min = data.groupby('customer_id')['temperature'].min().reset_index(
        name='temperature_Min')
    temperature_Max = data.groupby('customer_id')['temperature'].max().reset_index(
        name='temperature_Max')

    # print(temperature_Avg.head())
    # print(temperature_Min.head())
    # print(temperature_Max.head())
    # print(temperature_Avg.shape)
    # print(temperature_Min.shape)
    # print(temperature_Max.shape)

    NewData = pd.merge(NewData, temperature_Avg, on='customer_id')
    NewData = pd.merge(NewData, temperature_Min, on='customer_id')
    NewData = pd.merge(NewData, temperature_Max, on='customer_id')

    print('PhoneDataFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)

    return NewData


# PhoneDataFeatures()


def ReactivationsFeatures():
    data1 = pd.read_csv('./data/reactivations_ebb_set1.csv')
    data2 = pd.read_csv('./data/reactivations_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    del data['reactivation_date']

    # FeatureMap存放当前表所有特征的map关系
    FeatureMap = dict()
    featuresMapPath = './FeatureMap/ReactivationsFeatures.json'

    features = list(set(data['reactivation_channel'].tolist()))
    # print('features:', features)
    NewData = data.groupby('customer_id')['reactivation_channel'].apply(lambda x: (x == features[0]).sum()).reset_index(
        name=f'{features[0]}')

    for i in range(1, len(features)):
        CurrData = data.groupby('customer_id')['reactivation_channel'].apply(
            lambda x: (x == features[i]).sum()).reset_index(name=f'{features[i]}')
        # print('CurrData\n', CurrData.head())
        NewData = pd.merge(NewData, CurrData, on='customer_id')

    print('ReactivationsFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)

    # 保存每个特征的map
    FeatureMap['reactivation_channel'] = features
    # 把FeatureMap写入对应位置json
    writeFeatureMap(featuresMapPath, FeatureMap)
    return NewData


# ReactivationsFeatures()

def ReactivationsFeatures_eval():
    data = pd.read_csv('./data/reactivations_eval_set.csv')
    del data['reactivation_date']

    ReactivationsFeatureMap = dict()
    with open('./FeatureMap/ReactivationsFeatures.json') as file:
        ReactivationsFeatureMap = json.load(file)
    reactivation_channelFeatureList = ReactivationsFeatureMap['reactivation_channel']
    # print('reactivation_channelFeatureList', reactivation_channelFeatureList)

    NewData = data.groupby('customer_id')['reactivation_channel'].apply(
        lambda x: (x == reactivation_channelFeatureList[0]).sum()).reset_index(
        name=f'{reactivation_channelFeatureList[0]}')

    for i in range(1, len(reactivation_channelFeatureList)):
        CurrData = data.groupby('customer_id')['reactivation_channel'].apply(
            lambda x: (x == reactivation_channelFeatureList[i]).sum()).reset_index(
            name=f'{reactivation_channelFeatureList[i]}')
        # print('CurrData\n', CurrData.head())
        NewData = pd.merge(NewData, CurrData, on='customer_id')

    print('ReactivationsFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    # (10650, 14), 14维度
    return NewData


# ReactivationsFeatures_eval()

def RedemptionsFeatures():
    data1 = pd.read_csv('./data/redemptions_ebb_set1.csv')
    data2 = pd.read_csv('./data/redemptions_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)

    revenues_Avg = data.groupby('customer_id')['revenues'].mean().reset_index(
        name='revenues_Avg')
    revenues_Min = data.groupby('customer_id')['revenues'].min().reset_index(
        name='revenues_Min')
    revenues_Max = data.groupby('customer_id')['revenues'].max().reset_index(
        name='revenues_Max')

    # print(revenues_Avg.head())
    # print(revenues_Min.head())
    # print(revenues_Max.head())
    # print(revenues_Avg.shape)
    # print(revenues_Min.shape)
    # print(revenues_Max.shape)

    NewData = revenues_Avg
    NewData = pd.merge(NewData, revenues_Min, on='customer_id')
    NewData = pd.merge(NewData, revenues_Max, on='customer_id')

    print('RedemptionsFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


def RedemptionsFeatures_eval():
    data = pd.read_csv('./data/redemptions_eval_set.csv')
    revenues_Avg = data.groupby('customer_id')['revenues'].mean().reset_index(
        name='revenues_Avg')
    revenues_Min = data.groupby('customer_id')['revenues'].min().reset_index(
        name='revenues_Min')
    revenues_Max = data.groupby('customer_id')['revenues'].max().reset_index(
        name='revenues_Max')

    # print(revenues_Avg.head())
    # print(revenues_Min.head())
    # print(revenues_Max.head())
    # print(revenues_Avg.shape)
    # print(revenues_Min.shape)
    # print(revenues_Max.shape)

    NewData = revenues_Avg
    NewData = pd.merge(NewData, revenues_Min, on='customer_id')
    NewData = pd.merge(NewData, revenues_Max, on='customer_id')

    print('RedemptionsFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


def SupportFeatures():
    data1 = pd.read_csv('./data/support_ebb_set1.csv')
    data2 = pd.read_csv('./data/support_ebb_set1.csv')
    data = pd.concat([data1, data2], ignore_index=True)

    # FeatureMap存放当前表所有特征的map关系
    FeatureMap = dict()
    featuresMapPath = './FeatureMap/SupportFeatures.json'

    case_type_features = list(set(data['case_type'].tolist()))
    case_type_featuresMap = {case_type_features[i]: i for i in range(len(case_type_features))}
    # print('features', case_type_features)
    # print('featuresMap', case_type_featuresMap)
    NewData = data.groupby('customer_id')['case_type'].apply(lambda x: (x == case_type_features[0]).sum()).reset_index(
        name=f'{case_type_features[0]}')
    for i in range(1, len(case_type_features)):
        CurrData = data.groupby('customer_id')['case_type'].apply(
            lambda x: (x == case_type_features[i]).sum()).reset_index(name=f'{case_type_features[i]}')
        NewData = pd.merge(NewData, CurrData, on='customer_id')
    print('SupportFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)

    # 保存每个特征的map
    FeatureMap['case_type'] = case_type_featuresMap
    # 把FeatureMap写入对应位置json
    writeFeatureMap(featuresMapPath, FeatureMap)
    return NewData


# SupportFeatures()

def SupportFeatures_eval():
    data = pd.read_csv('./data/support_eval_set.csv')
    # SupportFeatureMap存放当前表所有特征的map关系
    SupportFeatureMap = dict()
    with open('./FeatureMap/SupportFeatures.json') as file:
        SupportFeatureMap = json.load(file)
    case_type_FeatureMap = SupportFeatureMap['case_type']
    case_type_Featurelist = list(case_type_FeatureMap.keys())
    # 根据train有的特征case_type_Featurelist，对eval进行聚类加和
    NewData = data.groupby('customer_id')['case_type'].apply(
        lambda x: (x == case_type_Featurelist[0]).sum()).reset_index(
        name=f'{case_type_Featurelist[0]}')
    for i in range(1, len(case_type_Featurelist)):
        CurrData = data.groupby('customer_id')['case_type'].apply(
            lambda x: (x == case_type_Featurelist[i]).sum()).reset_index(name=f'{case_type_Featurelist[i]}')
        NewData = pd.merge(NewData, CurrData, on='customer_id')
    print('SupportFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    #  (18901, 60)， 60维度特征
    return NewData


# SupportFeatures_eval()
# Suspensions停卡
def SuspensionsFeatures():
    # 每行end_date - start_date = 得到相差天数
    # 获得sum， avg, min,max
    data1 = pd.read_csv('./data/suspensions_ebb_set1.csv')
    data2 = pd.read_csv('./data/suspensions_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)

    # datadiff = pd.to_datetime(data['end_date']) - pd.to_datetime(data['start_date'])
    data['datadiff'] = pd.to_datetime(data['end_date']) - pd.to_datetime(data['start_date'])
    data['datadiff'] = (data['datadiff']).dt.days

    SuspensionsDay_Sum = data.groupby('customer_id')['datadiff'].sum().reset_index(
        name='SuspensionsDay_Sum')
    SuspensionsDay_Avg = data.groupby('customer_id')['datadiff'].mean().reset_index(
        name='SuspensionsDay_Avg')
    SuspensionsDay_Min = data.groupby('customer_id')['datadiff'].min().reset_index(
        name='SuspensionsDay_Min')
    SuspensionsDay_Max = data.groupby('customer_id')['datadiff'].max().reset_index(
        name='SuspensionsDay_Max')

    # print(SuspensionsDay_Sum.head())
    # print(SuspensionsDay_Avg.head())
    # print(SuspensionsDay_Min.head())
    # print(SuspensionsDay_Max.head())
    #
    # print(SuspensionsDay_Sum.shape)
    # print(SuspensionsDay_Avg.shape)
    # print(SuspensionsDay_Min.shape)
    # print(SuspensionsDay_Max.shape)

    NewData = SuspensionsDay_Sum
    NewData = pd.merge(NewData, SuspensionsDay_Avg, on='customer_id')
    NewData = pd.merge(NewData, SuspensionsDay_Min, on='customer_id')
    NewData = pd.merge(NewData, SuspensionsDay_Max, on='customer_id')

    print('SuspensionsFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# SuspensionsFeatures()

def SuspensionsFeatures_eval():
    # 每行end_date - start_date = 得到相差天数
    # 获得sum， avg, min,max
    data = pd.read_csv('./data/suspensions_eval_set.csv')

    # datadiff = pd.to_datetime(data['end_date']) - pd.to_datetime(data['start_date'])
    data['datadiff'] = pd.to_datetime(data['end_date']) - pd.to_datetime(data['start_date'])
    data['datadiff'] = (data['datadiff']).dt.days

    SuspensionsDay_Sum = data.groupby('customer_id')['datadiff'].sum().reset_index(
        name='SuspensionsDay_Sum')
    SuspensionsDay_Avg = data.groupby('customer_id')['datadiff'].mean().reset_index(
        name='SuspensionsDay_Avg')
    SuspensionsDay_Min = data.groupby('customer_id')['datadiff'].min().reset_index(
        name='SuspensionsDay_Min')
    SuspensionsDay_Max = data.groupby('customer_id')['datadiff'].max().reset_index(
        name='SuspensionsDay_Max')

    # print(SuspensionsDay_Sum.head())
    # print(SuspensionsDay_Avg.head())
    # print(SuspensionsDay_Min.head())
    # print(SuspensionsDay_Max.head())
    #
    # print(SuspensionsDay_Sum.shape)
    # print(SuspensionsDay_Avg.shape)
    # print(SuspensionsDay_Min.shape)
    # print(SuspensionsDay_Max.shape)

    NewData = SuspensionsDay_Sum
    NewData = pd.merge(NewData, SuspensionsDay_Avg, on='customer_id')
    NewData = pd.merge(NewData, SuspensionsDay_Min, on='customer_id')
    NewData = pd.merge(NewData, SuspensionsDay_Max, on='customer_id')

    print('SuspensionsFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# Throttling节流, 应该关注节流天数
def ThrottlingFeatures():
    data1 = pd.read_csv('./data/throttling_ebb_set1.csv')
    data2 = pd.read_csv('./data/throttling_ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)

    NewData = data.groupby('customer_id').size().reset_index(name='throttled_days')
    print('ThrottlingFeatures\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


# ThrottlingFeatures()

def ThrottlingFeatures_eval():
    data = pd.read_csv('./data/throttling_eval_set.csv')
    NewData = data.groupby('customer_id').size().reset_index(name='throttled_days')
    print('ThrottlingFeatures_eval\n', NewData.head())
    print('Shape\n', NewData.shape)
    return NewData


def MainFeatures():
    data1 = pd.read_csv('./data/ebb_set1.csv')
    data2 = pd.read_csv('./data/ebb_set2.csv')
    data = pd.concat([data1, data2], ignore_index=True)

    # FeatureMap存放当前表所有特征的map关系
    FeatureMap = dict()
    featuresMapPath = './FeatureMap/MainFeatures.json'

    # # last_plan_name_coded
    # last_plan_name_coded_features = list(set(data['last_plan_name_coded'].tolist()))
    # last_plan_name_coded_featuresMap = {last_plan_name_coded_features[i]: i for i in
    #                                     range(len(last_plan_name_coded_features))}
    # # print('last_plan_name_coded_featuresMap:', last_plan_name_coded_featuresMap)
    # data["last_plan_name_coded"].replace(last_plan_name_coded_featuresMap, inplace=True)

    # last_redemption_date
    del data['last_redemption_date']

    # first_activation_date
    del data['first_activation_date']

    # total_redemptions
    # reserve

    # tenure 使用期
    # reserve

    # number_upgrades
    # reserve

    # year
    del data['year']

    # manufacturer
    manufacturer_features = list(set(data['manufacturer'].tolist()))
    manufacturer_featuresMap = {manufacturer_features[i]: i for i in
                                range(len(manufacturer_features))}
    # print('manufacturer_featuresMap:', manufacturer_featuresMap)
    data["manufacturer"].replace(manufacturer_featuresMap, inplace=True)

    # operating_system
    # 1.tolowercase and covert non to None
    data['operating_system'] = data['operating_system'].str.lower()
    data['operating_system'] = data['operating_system'].where(pd.notnull(data['operating_system']), None)
    operating_system_features = list(set(data['operating_system'].tolist()))
    # 2.makemap
    operating_system_NameMap = dict()
    for featurename in operating_system_features:
        featurename = str(featurename)
        if featurename is not None:
            if 'ios' in featurename:
                operating_system_NameMap[featurename] = 'ios'
            elif 'android' in featurename:
                operating_system_NameMap[featurename] = 'android'
            elif 'proprietary' in featurename:
                operating_system_NameMap[featurename] = 'proprietary'
            elif 'windows' in featurename:
                operating_system_NameMap[featurename] = 'windows'
            elif 'blackberry' in featurename:
                operating_system_NameMap[featurename] = 'blackberry'
            else:
                operating_system_NameMap[featurename] = None
        else:
            operating_system_NameMap[featurename] = None
    # print('operating_system_NameMap:', operating_system_NameMap)
    data["operating_system"].replace(operating_system_NameMap, inplace=True)
    New_operating_system_features = list(set(data['operating_system'].tolist()))
    operating_system_featuresMap = {New_operating_system_features[i]: i for i in
                                    range(len(New_operating_system_features))}
    # print('operating_system_featuresMap:', operating_system_featuresMap)
    data["operating_system"].replace(operating_system_featuresMap, inplace=True)

    # language_preference
    del data['language_preference']

    # opt_out_email
    del data['opt_out_email']

    # opt_out_loyalty_email
    del data['opt_out_loyalty_email']

    # opt_out_loyalty_sms
    del data['opt_out_loyalty_sms']

    # 留################################################################################
    # opt_out_mobiles_ads
    del data['opt_out_mobiles_ads']

    # opt_out_phone
    del data['opt_out_phone']

    # state
    state_features = list(set(data['state'].tolist()))
    state_featuresMap = {state_features[i]: i for i in
                         range(len(state_features))}
    # print('state_featuresMap:', state_featuresMap)
    data["state"].replace(state_featuresMap, inplace=True)

    # total_revenues_bucket bucket越多，客户开通服务越多
    # reserve

    # marketing_comms_1
    del data['marketing_comms_1']

    # marketing_comms_2
    del data['marketing_comms_2']

    print('MainFeatures\n', data.head())
    print('Shape\n', data.shape)

    # 保存每个特征的map
    # FeatureMap['last_plan_name_coded'] = last_plan_name_coded_featuresMap
    FeatureMap['manufacturer'] = manufacturer_featuresMap
    FeatureMap['operating_system'] = operating_system_featuresMap
    FeatureMap['state'] = state_featuresMap
    # 把FeatureMap写入对应位置json
    writeFeatureMap(featuresMapPath, FeatureMap)
    return data


# data = MainFeatures()
# print(data)

def MainFeatures_eval():
    data = pd.read_csv('./data/eval_set.csv')
    FeaturesMap = dict()
    with open('./FeatureMap/MainFeatures.json') as file:
        FeaturesMap = json.load(file)

    # # last_plan_name_coded
    # last_plan_name_coded_featuresMap = FeaturesMap['last_plan_name_coded']
    # data["last_plan_name_coded"].replace(last_plan_name_coded_featuresMap, inplace=True)
    # last_plan_name_coded_features = list(set(data['last_plan_name_coded'].tolist()))
    # # print('Eval last_plan_name_coded_features', last_plan_name_coded_features)

    # last_redemption_date
    del data['last_redemption_date']

    # first_activation_date
    del data['first_activation_date']

    # total_redemptions
    # reserve

    # tenure 使用期
    # reserve

    # number_upgrades
    # reserve

    # year
    del data['year']

    # manufacturer
    manufacturer_featuresMap = FeaturesMap['manufacturer']
    manufacturer_features = list(set(data['manufacturer'].tolist()))
    # print('Eval manufacturer_features', manufacturer_features)
    # Eval有但是训练集没有，那么要添加
    for feature in list(set(manufacturer_features) - manufacturer_featuresMap.keys()):
        manufacturer_featuresMap[feature] = len(manufacturer_featuresMap) + 1
    data["manufacturer"].replace(manufacturer_featuresMap, inplace=True)
    manufacturer_features = list(set(data['manufacturer'].tolist()))
    # print('Eval manufacturer_features', manufacturer_features)

    # operating_system
    # 1.tolowercase and covert non to None
    operating_system_featuresMap = FeaturesMap['operating_system']
    # 替换'null'为None,否则识别出错
    if 'null' in operating_system_featuresMap.keys():
        value = operating_system_featuresMap['null']
        del operating_system_featuresMap['null']
        operating_system_featuresMap[None] = value
    # print('operating_system_featuresMap', operating_system_featuresMap)
    data['operating_system'] = data['operating_system'].str.lower()
    data['operating_system'] = data['operating_system'].where(pd.notnull(data['operating_system']), None)
    operating_system_features = list(set(data['operating_system'].tolist()))
    # 2.makemap
    operating_system_NameMap = dict()
    for featurename in operating_system_features:
        featurename = str(featurename)
        if featurename is not None:
            if 'ios' in featurename:
                operating_system_NameMap[featurename] = 'ios'
            elif 'android' in featurename:
                operating_system_NameMap[featurename] = 'android'
            elif 'proprietary' in featurename:
                operating_system_NameMap[featurename] = 'proprietary'
            elif 'windows' in featurename:
                operating_system_NameMap[featurename] = 'windows'
            elif 'blackberry' in featurename:
                operating_system_NameMap[featurename] = 'blackberry'
            else:
                operating_system_NameMap[featurename] = None
        else:
            operating_system_NameMap[featurename] = None
    # print('operating_system_NameMap:', operating_system_NameMap)
    # 先替换成分类后的名字
    data["operating_system"].replace(operating_system_NameMap, inplace=True)
    # New_operating_system_features = list(set(data['operating_system'].tolist()))
    # print(New_operating_system_features)
    # 再把分类的名字替换成int
    data["operating_system"].replace(operating_system_featuresMap, inplace=True)
    # New_operating_system_features = list(set(data['operating_system'].tolist()))
    # print(New_operating_system_features)

    # language_preference
    del data['language_preference']

    # opt_out_email
    del data['opt_out_email']

    # opt_out_loyalty_email
    del data['opt_out_loyalty_email']

    # opt_out_loyalty_sms
    del data['opt_out_loyalty_sms']

    # opt_out_mobiles_ads
    del data['opt_out_mobiles_ads']

    # opt_out_phone
    del data['opt_out_phone']

    # state
    state_features_json = FeaturesMap['state']
    # 替换'null'为None,否则识别出错
    if 'NaN' in state_features_json.keys():
        value = state_features_json['NaN']
        del state_features_json['NaN']
        state_features_json[None] = value
    # print('state_features_json', state_features_json)
    data['state'] = data['state'].where(pd.notnull(data['state']), None)
    data["state"].replace(state_features_json, inplace=True)
    # state_features = list(set(data['state'].tolist()))
    # print('state_features', state_features)
    # total_revenues_bucket
    # reserve

    # marketing_comms_1
    del data['marketing_comms_1']

    # marketing_comms_2
    del data['marketing_comms_2']

    print('MainFeatures_eval\n', data.head())
    print('Shape\n', data.shape)
    # train有10维度
    # eval有9维度，少1维label
    return data


# MainFeatures_eval()
########################################TrainFeature###############################################
def Build_TrainFeatures():
    # Create FeatureMap and Features dictionary
    FeatureMap, Features = os.path.exists('./FeatureMap'), os.path.exists('./Features')
    if not FeatureMap:
        os.makedirs('./FeatureMap')
    if not Features:
        os.makedirs('./Features')

    main_feature = MainFeatures()
    activation_feature = ActivationFeature()
    deactivations_feature = DeactivationsFeature()
    interactions_feature = InteractionsFeature()
    IVRCalls_features = IVRCallsFeatures()
    loyaltyprogram_features = LoyaltyprogramFeatures()
    network_feature = NetworkFeatures()
    phoneData_features = PhoneDataFeatures()
    reactivations_features = ReactivationsFeatures()
    redemptions_features = RedemptionsFeatures()
    support_features = SupportFeatures()
    suspensions_features = SuspensionsFeatures()
    throttling_features = ThrottlingFeatures()

    Features = pd.merge(main_feature, activation_feature, on='customer_id', how='left')
    Features = pd.merge(Features, deactivations_feature, on='customer_id', how='left')
    Features = pd.merge(Features, interactions_feature, on='customer_id', how='left')
    Features = pd.merge(Features, IVRCalls_features, on='customer_id', how='left')
    Features = pd.merge(Features, loyaltyprogram_features, on='customer_id', how='left')
    Features = pd.merge(Features, network_feature, on='customer_id', how='left')
    Features = pd.merge(Features, phoneData_features, on='customer_id', how='left')
    Features = pd.merge(Features, reactivations_features, on='customer_id', how='left')
    Features = pd.merge(Features, redemptions_features, on='customer_id', how='left')
    Features = pd.merge(Features, support_features, on='customer_id', how='left')
    Features = pd.merge(Features, suspensions_features, on='customer_id', how='left')
    Features = pd.merge(Features, throttling_features, on='customer_id', how='left')
    print('TrainFeatures\n', Features.head())
    print('Shape\n', Features.shape)
    Features.to_csv('./Features/TrainFeature.csv')
    return Features


####################################################Eval Feature##################################
def Build_EvalFeatures():
    main_feature_eval = MainFeatures_eval()
    activation_feature_eval = ActivationFeature_eval()
    deactivations_feature_eval = DeactivationsFeature_eval()
    interactions_feature_eval = InteractionsFeature_eval()
    IVRCalls_features_eval = IVRCallsFeatures_eval()
    loyaltyprogram_features_eval = LoyaltyprogramFeatures_eval()
    network_feature_eval = NetworkFeatures_eval()
    phoneData_features_eval = PhoneDataFeatures_eval()
    reactivations_features_eval = ReactivationsFeatures_eval()
    redemptions_features_eval = RedemptionsFeatures_eval()
    support_features_eval = SupportFeatures_eval()
    suspensions_features_eval = SuspensionsFeatures_eval()
    throttling_features_eval = ThrottlingFeatures_eval()

    Features = pd.merge(main_feature_eval, activation_feature_eval, on='customer_id', how='left')
    Features = pd.merge(Features, deactivations_feature_eval, on='customer_id', how='left')
    Features = pd.merge(Features, interactions_feature_eval, on='customer_id', how='left')
    Features = pd.merge(Features, IVRCalls_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, loyaltyprogram_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, network_feature_eval, on='customer_id', how='left')
    Features = pd.merge(Features, phoneData_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, reactivations_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, redemptions_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, support_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, suspensions_features_eval, on='customer_id', how='left')
    Features = pd.merge(Features, throttling_features_eval, on='customer_id', how='left')
    print('EvalFeature\n', Features.head())
    print('Shape\n', Features.shape)
    Features.to_csv('./Features/EvalFeature.csv')
    return Features


#########################################特征选择################################################
def FeatureSelection():
    dataset = pd.read_csv('./Features/TrainFeature.csv', index_col=[0])
    dataset_label = dataset['ebb_eligible'].tolist()
    dataset_customer_id = dataset['customer_id']

    del dataset['ebb_eligible']
    del dataset['customer_id']

    # print('dataset_label', dataset_label.tolist())
    evalset = pd.read_csv('./Features/EvalFeature.csv', index_col=[0])
    evalset_customer_id = evalset['customer_id']
    del evalset['customer_id']
    # print('dataset Shape', dataset.shape)
    # print('dataset columns', list(dataset.columns))

    # 标记特征是category还是float
    #   1.Category类型，只有['last_plan_name_coded', 'manufacturer', 'operating_system','state']
    #   2.数值类型，int float，剩余其他feature
    CategoryFeature = {'last_plan_name_coded', 'manufacturer', 'operating_system', 'state'}
    FloatFeature = set(set(list(dataset.columns)) - CategoryFeature)

    # 1.删除None占36%以上的feature,剩余109维
    dataset.dropna(thresh=0.25 * dataset.shape[0], axis=1, inplace=True)
    # dataset中保留的feature
    dataset_features = list(dataset.columns)
    evalset = evalset[dataset_features]
    print('dataset Shape after dropna', dataset.shape)
    print('evalset Shape after dropna', evalset.shape)

    # 2.填充空值
    #   1.Category类型填充方法:众数 mode()
    #   2.数值类型填充方法:中位数 median 或者 均值mean
    #   3.特殊方法：Category类型用classfier分类， 数值类型用linear回归
    #   2.1填充dataset
    for featureName in dataset_features:
        if featureName != 'customer_id':
            # 众数填CategoryFeature
            if featureName in CategoryFeature:
                dataset[featureName].fillna(dataset[featureName].mode()[0], inplace=True)
            # 中位数填数值类型Feature
            else:
                dataset[featureName].fillna(dataset[featureName].median(), inplace=True)

    #   2.2填充evalset
    for featureName in dataset_features:
        if featureName != 'customer_id':
            # 众数填CategoryFeature
            if featureName in CategoryFeature:
                evalset[featureName].fillna(evalset[featureName].mode()[0], inplace=True)
            # 中位数填数值类型Feature
            else:
                evalset[featureName].fillna(evalset[featureName].median(), inplace=True)

    # 先进性Normalizer(MinMaxScaler)，再用方差筛选
    # Normalizer后的dataset和evalset不用作下一步训练（仅用来筛选特征）
    # Norm_transform = Normalizer()
    Norm_transform = MinMaxScaler()
    dataset_norm = Norm_transform.fit_transform(dataset)
    evalset_norm = Norm_transform.transform(evalset)
    dataset_norm = pd.DataFrame(dataset_norm, columns=dataset.columns)
    # evalset_norm = pd.DataFrame(evalset_norm, columns=dataset.columns)

    # 3.删除方差低底的feature
    # 方差代表feature内部的差异，如果方差为0，意味着feature内部全部为一个相同的数，那么这个feature对于聚类和分类没有意义
    # 3.0计算每个特征的方差
    FeatureVar = dict()  # 存放特征:方差
    for featureName in dataset_features:
        FeatureVar[featureName] = dataset_norm[featureName].var().round(4)
    # print(FeatureVar)

    # 可选项1：设置FeatureThreshold，筛选Feature ############################################################################
    # 用下0.75分卫数作为FeatureThreshold，低于FeatureThreshold的特征舍弃，高于的保留
    # 共0-19
    FeatureThresholdList = np.quantile(list(FeatureVar.values()),
                                       [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                                        0.80, 0.85, 0.9,
                                        0.95, 1])
    # print('FeatureThresholdList', FeatureThresholdList)
    FeatureThreshold = FeatureThresholdList[13]
    # print('FeatureThreshold', FeatureThreshold)
    # 移除低于FeatureThreshold的特征
    removed_feature = [feature for feature in dataset_features if FeatureVar[feature] < FeatureThreshold]

    # 3.1删除dataset方差底的特征
    # 3.2根据dataset保留的特征，evalset保留相同的特征
    for featureName in removed_feature:
        del dataset[featureName]
        del evalset[featureName]

    # 更新保留的dataset_features
    dataset_features = list(dataset.columns)

    # 同一量纲化StandardScaler
    ss = StandardScaler()
    dataset_ss = ss.fit_transform(dataset)
    dataset = pd.DataFrame(dataset_ss, columns=dataset_features)
    evalset_ss = ss.transform(evalset)
    evalset = pd.DataFrame(evalset_ss, columns=dataset_features)

    print('Features after Var Threshold:', list(dataset.columns))
    print('Number of Features', len(dataset.columns))

    # # PCA
    # # n_components=20
    # print('PCA')
    # pca = PCA(n_components=20)
    # dataset_pca = pca.fit_transform(dataset)
    # evalset_pca = pca.transform(evalset)
    #
    # dataset = pd.DataFrame(dataset_pca, columns=[f'Feature{i}' for i in range(len(dataset_pca[0]))])
    # evalset = pd.DataFrame(evalset_pca, columns=[f'Feature{i}' for i in range(len(dataset_pca[0]))])
    #
    # print('Features after PCA:', list(dataset.columns))
    # print('Number of Features', len(dataset.columns))

    # 保存数据
    dataset['customer_id'], evalset['customer_id'] = dataset_customer_id, evalset_customer_id
    dataset['ebb_eligible'] = dataset_label

    dataset.to_csv('./Features/TrainFeatureSelected.csv')
    evalset.to_csv('./Features/EvalFeatureSelected.csv')
    print('Trainset write to ./Features/TrainFeatureSelected.csv')
    print('Evalset write to ./Features/EvalFeatureSelected.csv')

    print('Dataset Shape', dataset.shape)
    print('EvalSet Shape', evalset.shape)

    return dataset, evalset




# Build_TrainFeatures()
# Build_EvalFeatures()

FeatureSelection()
