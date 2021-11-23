import pandas as pd
train_data = pd.read_csv('raw_data/train_public.csv')
test_data = pd.read_csv('raw_data/test_public.csv')
sub=pd.read_csv("nn2.csv")
sub=sub.rename(columns={'id': 'loan_id'})
sub.loc[sub['isDefault']<0.498767,'isDefault'] = 0
nw_sub=sub[(sub['isDefault']==0)]
nw_test_data=test_data.merge(nw_sub,on='loan_id',how='inner')
nw_train_data = pd.concat([train_data,nw_test_data]).reset_index(drop=True)
nw_train_data.to_csv("raw_data/nw_train_public.csv",index=0)