import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import random
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import StratifiedKFold
from dateutil.relativedelta import relativedelta

train_data = pd.read_csv('raw_data/nw_train_public.csv')
submit_example = pd.read_csv('raw_data/submit_example.csv')
test_public = pd.read_csv('raw_data/test_public.csv')
train_inte = pd.read_csv('raw_data/train_internet.csv')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)


def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            tree_learner='serial',
            num_leaves=16,
            max_depth=4,
            learning_rate=0.01,
            n_estimators=10000,
            subsample=0.45,
            feature_fraction=0.5,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=2021,
            is_unbalance=True,
            num_thread=40,
            metric='auc'
        )

        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', early_stopping_rounds=40, verbose=100 # 30
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds

    return oof_preds, test_[['loan_id', 'isDefault']], feature_importance_df


def display_importances(feature_importance_df_):
    # Plot feature importances
    pass


def workYearDIc(x):
    if str(x) == 'nan':
        return -1
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'

random.seed(44444444)
def clear_early_return(temp):
    #k=temp.loc[:,['early_return', 'early_return_amount', 'early_return_amount_3mon']]
    for index in range(len(temp)):
        if temp.early_return[index] == 0 and temp.early_return_amount[index] > 0:
            temp.early_return[index] = random.randint(1,3)
        if temp.early_return[index] == 0 and temp.early_return_amount_3mon[index] > 0:
            temp.early_return[index] = random.randint(1,3)
def clean_test(tmp):
    for index in range(len(tmp)):
        if tmp.early_return[index] != 0 and tmp.early_return_amount[index] == 0:
            tmp.early_return_amount[index]=random.uniform(1000,tmp.total_loan[index])


clear_early_return(train_data)
clear_early_return(test_public)
clear_early_return(train_inte)
clean_test(test_public)

class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}
train_data["pub_dero_bankrup"] = train_data["pub_dero_bankrup"].fillna(train_data["pub_dero_bankrup"].median())
timeMax = pd.to_datetime('1-Dec-21')
train_data['work_year'] = train_data['work_year'].map(workYearDIc)
test_public['work_year'] = test_public['work_year'].map(workYearDIc)
train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)
# f=['f0','f1','f3','f4','f2']
# train_data[f].fillna(0)
# train_inte[f].fillna(0)
# test_public[f].fillna(0)
train_data["pub_dero_bankrup"] = train_data["pub_dero_bankrup"].fillna(train_data["pub_dero_bankrup"].median())
train_data['pro']=train_data['interest']*train_data['year_of_loan']
test_public['pro']=test_public['interest']*test_public['year_of_loan']
train_inte['pro']=train_inte['interest']*train_inte['year_of_loan']
train_data['loan_year']=train_data['total_loan']/train_data['year_of_loan']
test_public['loan_year']=test_public['total_loan']/test_public['year_of_loan']
train_inte['loan_year']=train_inte['total_loan']/train_inte['year_of_loan']
train_data['early_ratio']=train_data['early_return_amount']/train_data['total_loan']
test_public['early_ratio']=test_public['early_return_amount']/test_public['total_loan']
train_inte['early_ratio']=train_inte['early_return_amount']/train_inte['total_loan']
train_data['early_times_ratio']=train_data['early_return']/train_data['year_of_loan']
test_public['early_times_ratio']=test_public['early_return']/test_public['year_of_loan']
train_inte['early_times_ratio']=train_inte['early_return']/train_inte['year_of_loan']
train_data['recircle_ratio']=train_data['recircle_b']/train_data['total_loan']
test_public['recircle_ratio']=test_public['recircle_b']/test_public['total_loan']
train_inte['recircle_ratio']=train_inte['recircle_b']/train_inte['total_loan']
train_data['recircle_amt']=train_data['recircle_u']*train_data['total_loan']
test_public['recircle_amt']=test_public['recircle_u']*test_public['total_loan']
train_inte['recircle_amt']=train_inte['recircle_u']*train_inte['total_loan']

train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
test_public['earlies_credit_mon'] = pd.to_datetime(test_public['earlies_credit_mon'].map(findDig))
train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)
train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

train_data['issue_date_month'] = train_data['issue_date'].dt.month
test_public['issue_date_month'] = test_public['issue_date'].dt.month
train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
test_public['issue_date_dayofweek'] = test_public['issue_date'].dt.dayofweek

train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
test_public['earliesCreditMon'] = test_public['earlies_credit_mon'].dt.month
train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
test_public['earliesCreditYear'] = test_public['earlies_credit_mon'].dt.year

###internet数据

train_inte['issue_date_month'] = train_inte['issue_date'].dt.month
train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek
train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month
train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year

cat_cols = ['employer_type', 'industry']

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])

    # Internet处理
    train_inte[col] = lbl.transform(train_inte[col])

# 'f1','policy_code','app_type' 这三个去掉是881
# ,'f1','policy_code','app_type'
col_to_drop = ['issue_date', 'earlies_credit_mon','policy_code']
train_data = train_data.drop(col_to_drop, axis=1)
test_public = test_public.drop(col_to_drop, axis=1)

##internet处理
train_inte = train_inte.drop(col_to_drop, axis=1)
# 暂时不变
# train_inte = train_inte.rename(columns={'is_default':'isDefault'})
# data = pd.concat( [train_data,test_public] )
tr_cols = set(train_data.columns)
same_col = list(tr_cols.intersection(set(train_inte.columns)))
train_inteSame = train_inte[same_col].copy()

Inte_add_cos = list(tr_cols.difference(set(same_col)))
for col in Inte_add_cos:
    train_inteSame[col] = np.nan

# 81后加
# for col in cat_cols:
#     dum = pd.get_dummies(data[col], prefix='OneHot_'+col +'_')
#     data = pd.concat([data, dum], axis=1)
# #     del data[col]
#     del dum

y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)

IntePre['isDef'] = train_inte['is_default']
from sklearn.metrics import roc_auc_score

roc_auc_score(IntePre['isDef'], IntePre.isDefault)
## 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
InteId = IntePre.loc[IntePre.isDefault < 0.5 , 'loan_id'].tolist()

train_data['dataSourse'] = 1
test_public['dataSourse'] = 1
train_inteSame['dataSourse'] = 0
train_inteSame['isDefault'] = train_inte['is_default']
use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)

# InteId = IntePre.loc[IntePre.isDefault<0.05, 'loan_id'].tolist()
# train_inte = train_inte.rename(columns={'is_default':'isDefault'})

# train_data['dataSourse'] = 1
# test_public['dataSourse'] = 1
# train_inte['dataSourse'] = 0


# use_te = train_inte[train_inte.loan_id.isin( InteId )].copy()
# data = pd.concat([ train_data,test_public,use_te]).reset_index(drop=True)

# IntePre.isDefault
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'], color="black", kde=True, bins=120, label='train_data')
# sns.distplot(train_inte[col],color="red", kde=True,bins=120, label='train_inte')
plt.legend();
plt.show()
train = data[data['isDefault'].notna()]
test = data[data['isDefault'].isna()]
# for col in ['sub_class', 'work_type']:
#     del train[col]
#     del test[col]

del data
del train_data, test_public

y = train['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds, test_preds, importances = train_model(train, test, y, folds)
test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('baseline891.csv', index=False)