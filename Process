mport numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
import re
from sklearn.metrics import mean_squared_error
import logging
train = pd.read_csv('.../jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('.../jinnan_round1_testB_20190121.csv', encoding = 'gb18030')
#########
# 删除类别唯一的特征Percentage of values in the biggest category=1
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
# 删除某一类别占比超过90%的列
good_cols = list(train.columns) #将train中特征以列表形式表示
# print(good_cols)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.90:
        good_cols.remove(col)
        # print(col, rate)
# print(good_cols)

# 暂时不删除，后面构造特征需要
good_cols.append('A1')
good_cols.append('A3')
good_cols.append('A4')
########
target_col = "收率"
# 删除异常值
# print(train[train['收率'] < 0.87])

train = train[train['收率'] > 0.87]
train.loc[train['B14'] == 40, 'B14'] = 400
train = train[train['B14']>=400]

# 合并数据集, 顺便处理异常数据
target = train['收率']
train.loc[train['A25'] == '1900/3/10 0:00', 'A25'] = train['A25'].value_counts().values[0]
train['A25'] = train['A25'].astype(int)
train.loc[train['B14'] == 40, 'B14'] = 400
# test.loc[test['B14'] == 385, 'B14'] = 385

test_select = {}
for v in [280, 385, 390, 785]:
    # print(v)
    # print(test[test['B14'] == v]['样本id'])
    test_select[v] = test[test['B14'] == v]['样本id'].index
    # print(test[test['B14'] == v]['样本id'].index)
    # print(test_select[v])

del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)
def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0

    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600

    return tm

for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f, '应该在前面被删除了！')
def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm


for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)

data['样本id'] = data['样本id'].apply(lambda x: x.split('_')[1])
data['样本id'] = data['样本id'].astype(int)
########
categorical_columns = [f for f in data.columns if f not in ['样本id']]
# print(categorical_columns)
numerical_columns = [f for f in data.columns if f not in categorical_columns]
# print(numerical_columns)
#有风的冬老哥，在群里无意爆出来的特征，让我提升了三个个点，当然也可以顺此继续扩展
data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')
del data['A1']
del data['A3']
del data['A4']
categorical_columns.remove('A1')
categorical_columns.remove('A3')
categorical_columns.remove('A4')
train = data[:train.shape[0]]
test  = data[train.shape[0]:]

# ''''
# 添加新特征，将收率进行分箱，然后构造每个特征中的类别对应不同收率的均值
train['target'] = target
# print(train['target'])

train['intTarget'] = pd.cut(train['target'], 5, labels=False) # 需要将数据值分段并排序到bins中时使用cut。 此函数对于从连续变量转换为离散变量也很有用
# print(train['inTarget'])
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
mean_columns = []
for f1 in categorical_columns:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
            mean_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            train[col_name] = train['B14'].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean_columns.remove(col_name)
            else:
                test[col_name] = test['B14'].map(order_label)

train.drop(li + ['target'], axis=1, inplace=True)
##########



# 基本数据处理完毕, 开始拼接数据
# train = data[:train.shape[0]]
# test  = data[train.shape[0]:]

train['target'] = list(target)
new_train = train[mean_columns+numerical_columns].copy()
new_train['target']=list(target)
new_train = new_train.sort_values(['样本id'], ascending=True)
train_copy = train[mean_columns+numerical_columns].copy()
train_copy['target']=list(target)
train_copy = train_copy.sort_values(['样本id'], ascending=True)

# 把train加长两倍
train_len = len(new_train)
new_train = pd.concat([new_train, train_copy])

# 把加长两倍的train拼接到test后面
new_test = test[mean_columns+numerical_columns].copy()
new_test = pd.concat([new_test, new_train])

import sys
# 开始向后做差
diff_train = pd.DataFrame()
ids = list(train_copy['样本id'].values)
# print(ids)
from tqdm import tqdm
import os
# 构造新的训练集
if os.path.exists('C:/Users/whositer/Desktop/JinNan/newTry/diff_train-BF.csv'):
    diff_train = pd.read_csv('C:/Users/whositer/Desktop/JinNan/newTry/diff_train-BF.csv')
else:
    for i in tqdm(range(1, train_len)):
        # 分别间隔 -1, -2, ... -len行 进行差值,得到实验的所有对比实验
        diff_tmp = new_train.diff(-i)
        diff_tmp = diff_tmp[:train_len]
        diff_tmp.columns = [col_ + '_difference' for col_ in
                            diff_tmp.columns.values]
        # 求完差值后加上样本id
        diff_tmp['样本id'] = ids
        diff_train = pd.concat([diff_train, diff_tmp])

    diff_train.to_csv('C:/Users/whositer/Desktop/JinNan/newTry/diff_train-BF.csv', index=False)

# 构造新的测试集
diff_test = pd.DataFrame()
ids_test = list(test['样本id'].values)
test_len = len(test)
if os.path.exists('C:/Users/whositer/Desktop/JinNan/newTry/diff_test-BF.csv'):
    diff_test = pd.read_csv('C:/Users/whositer/Desktop/JinNan/newTry/diff_test-BF.csv')
else:
    for i in tqdm(range(test_len, test_len+train_len)):
        # 分别间隔 - test_len , -test_len -1 ,.... - test_len - train_len +1 进行差值, 得到实验的所有对比实验
        diff_tmp = new_test.diff(-i)
        diff_tmp = diff_tmp[:test_len]
        diff_tmp.columns = [col_ + '_difference' for col_ in
                            diff_tmp.columns.values]
        # 求完差值后加上样本id
        diff_tmp['样本id'] = ids_test
        diff_test = pd.concat([diff_test, diff_tmp])

    diff_test = diff_test[diff_train.columns]
    diff_test.to_csv('C:/Users/whositer/Desktop/JinNan/newTry/diff_test-BF.csv', index=False)


# print(train.columns.values)
# 和train顺序一致的target
train_target = train['target']
train.drop(['target'], axis=1, inplace=True)
# 拼接原始特征
diff_train = pd.merge(diff_train, train, how='left', on='样本id')
diff_test = pd.merge(diff_test, test, how='left', on='样本id')
target = diff_train['target_difference']
diff_train.drop(['target_difference'], axis=1, inplace=True)
diff_test.drop(['target_difference'], axis=1, inplace=True)

X_train = diff_train
y_train = target
X_test = diff_test
# print(X_train.columns.values)
#lightgbm训练过程
param = {'num_leaves': 31, #31
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         # "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l2": 0.1,
         # "lambda_l1": 0.1,
         "verbosity": -1}
groups = X_train['样本id'].values

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(diff_train))
predictions_lgb = np.zeros(len(diff_test))

feature_importance = pd.DataFrame()
feature_importance['feature_name'] = X_train.columns.values
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    dev = X_train.iloc[trn_idx]
    val = X_train.iloc[val_idx]

    trn_data = lgb.Dataset(dev, y_train.iloc[trn_idx])
    val_data = lgb.Dataset(val, y_train.iloc[val_idx])

    num_round = 6000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(val, num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
#########
#########
#########1-19晚上
xgb_params = {'eta': 0.005, 'max_depth': 9, 'subsample': 0.9, 'colsample_bytree': 0.8,'min_child_weight':6,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'learnig_rate':0.066,'gpu_id' : 0,'tree_method' : 'gpu_hist'}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(diff_train))
predictions_xgb = np.zeros(len(diff_test))
feature_importance2 = pd.DataFrame()
feature_importance2['feature_name'] = X_train.columns.values
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    # print("fold n°{}".format(fold_ + 1))
    print("fold n°{}".format(fold_ + 1))
    # print("fold n°{}".format(fold_ + 1))
    dev = X_train.iloc[trn_idx]
    val = X_train.iloc[val_idx]
    trn_data = xgb.DMatrix(dev, y_train[trn_idx])
    val_data = xgb.DMatrix(val, y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=4000, evals=watchlist, early_stopping_rounds=100,
                    verbose_eval=100, params=xgb_params)
    # oof_xgb[trn_idx] = clf.predict(xgb.DMatrix(X_train[trn_idx]), ntree_limit=clf.best_ntree_limit)
    # oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    # predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(val), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    # clf_1=LinearSVR()
    # clf_1.fit(trn_data,trn_y)
    # clf_2=RandomForestRegressor()
    # clf_2.fit(trn_data,trn_y)
    clf_3=BayesianRidge()
    clf_3.fit(trn_data,trn_y)
    # clf_4=XGBRegressor()
    # clf_4.fit(trn_data,trn_y)
    # clf_5=LGBMRegressor()
    # clf_5.fit(trn_data,trn_y)
    # oof_stack[trn_idx]=clf_3.predict(trn_data)
    oof_stack[val_idx] =clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
# 还原train target
diff_train['compare_id'] = diff_train['样本id'] - diff_train['样本id_difference']
train['compare_id'] = train['样本id']
train['compare_target'] = list(train_target)   #对比的样本
#把做差的target拼接回去
diff_train = pd.merge(diff_train, train[['compare_id', 'compare_target']], how='left', on='compare_id')
# print(diff_train.columns.values)
diff_train['pre_target_diff'] =oof_stack
diff_train['pre_target'] = diff_train['pre_target_diff'] + diff_train['compare_target']

mean_result = diff_train.groupby('样本id')['pre_target'].mean().reset_index(name='pre_target_mean')
true_result = train[['样本id', 'compare_target']]
mean_result = pd.merge(mean_result, true_result, how='left', on='样本id')

# pre_target = mean_result['pre_target_mean'].values
# true_target = mean_result['']
# 还原test target
diff_test['compare_id'] = diff_test['样本id'] - diff_test['样本id_difference']
diff_test = pd.merge(diff_test, train[['compare_id', 'compare_target']], how='left', on='compare_id')
diff_test['pre_target_diff'] = predictions
diff_test['pre_target'] = diff_test['pre_target_diff'] + diff_test['compare_target']

mean_result_test = diff_test.groupby(diff_test['样本id'], sort=False)['pre_target'].mean().reset_index(name='pre_target_mean')
print(mean_result_test)
test = pd.merge(test, mean_result_test, how='left', on='样本id')
sub_df = pd.read_csv('C:/Users/whositer/Desktop/JinNan/jinnan_round1_submit_20181227.csv', header=None)
sub_df[0]=test['样本id']
sub_df[1] = test['pre_target_mean']
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
for v in test_select.keys():
    if v == 280:
        x = 0.947
    elif v == 385 or v == 785:
        x = 0.879
    elif v == 390:
        x = 0.89
    sub_df.loc[test_select[v], 1] = x

sub_df.to_csv('.../jinnan_round_submit_BF.csv', index=False, header=False)
