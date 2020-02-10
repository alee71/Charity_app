# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:39:31 2020

@author: Andrea
"""

import pandas as pd 
import sqlalchemy as sa
from pathlib import Path
from sqlalchemy import create_engine
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import seaborn as sns
from pylab import rcParams
from collections import Counter
#%%
query=''' SELECT *
FROM Open990
'''
e = sa.create_engine(r'sqlite:///D:\Program_practices\Charity\open990database.db')
df= pd.read_sql_query(query, e)
irrelevant=['country', 'org_form.other_form_desc', 'grp_exempt_num', 'state_legaldomicile', 
           'grp_return', 'grp_subs_all_incl', 'exempt_status.501c3', 'exempt_status.501c_any',
           'exempt_status.501c_txt','income_tot_unrelated', 'income_net_unrelated', 
           'rev_giftsgrants_tot_prioryr', 'rev_prgmservice_prioryr', 'rev_investmt_prioryr',
           'rev_other_prioryr','fundraiseservfee_expense_tot_prioryr', 'grants_expense_tot_prioryr',
           'benefits_expense_tot_prioryr', 'salaries_expense_tot_prioryr', 'fundraise_expense_tot_curyr',
           'other_expense_tot_prioryr','rev_giftsgrants_tot_curyr', 'rev_prgmservice_curyr',
           'rev_investmt_curyr', 'rev_other_curyr', 'fundraiseservfee_expense_tot_curyr', 'grants_expense_tot_curyr',
           'benefits_expense_tot_curyr', 'salaries_expense_tot_curyr', 'fundraise_expense_tot_curyr',
           'other_expense_tot_curyr','operate_hospfacility', 'relationsamongemployees', 'delegate_management',
           'memb_or_stockholder', 'memb_elect_board','decisions_outside_board', 'govern_body_minutes',
           'committee_minutes', 'no_officer_address', 'policies_ref_chapt', '990_to_members', 'conflict_interest_policy',
           'conflict_disclosure','enforce_conflict_policy', 'whistleblower_policy', 'doc_retention_policy',
           'ceo_comp_review','officer_comp_review', 'in_joint_venture', 'joint_venture_policy', 
           'forms_ownwebsite', 'record_id', 'schema_version', 'irs_efile_id', 'random_string'
           ]
df=df.drop(columns=irrelevant)
w= sa.create_engine(r'sqlite:///D:\Program_practices\Charity\open990database.sqlite')
df.to_sql('Full_data', w, if_exists='replace', index=False)

#%%
query=''' SELECT *
FROM Full_Volunteer
GROUP BY name_org, tax_yr
'''
df_fullbackup = pd.read_sql_query(query, w)
# nj.to_sql('NJ', w, if_exists='replace', index=False)


#%%
df['volunteer_tot_ct']=df['volunteer_tot_ct'].fillna(0).astype(int)
df=df[df['volunteer_tot_ct'] !=0]
irrelevant=['tax_date_begin', 'tax_date_end', 'doing_business_as', 'phone', 'website', 
             'address', 'zip']
df=df.drop(columns=irrelevant)
df.to_sql('Full_Volunteer', w, if_exists='replace', index=False)
#%%
# query=''' SELECT y1.*
# FROM Full_Volunteer y1
# WHERE y1.tax_yr = (SELECT max(y2.tax_yr)
#                    FROM Full_Volunteer y2
#                    WHERE y2.ein=y1.ein)
# '''
# key = pd.read_sql_query(query, w)

# # df_temp=df.drop_duplicates().reset_index(drop=True)


#%%
df_backup=df.copy()
key=df[['ein', 'ntee_code_nccs', 'ntee_description_nccs']]
key=key.drop_duplicates().dropna().reset_index(drop=True)
# key = key[np.isfinite(key['ntee_code_nccs'])]
# key=key.drop_duplicates().dropna().reset_index(drop=True)
df=df.groupby('ein', group_keys=False).apply(lambda x: x.loc[x['tax_yr'].idxmax()])
df.index.name=None
df=pd.merge(df,key, on='ein', how='left')

# The duplicates are due to changes in business ntee code; we will take the newest code
df = df.drop_duplicates(subset='ein', keep="last")

df.to_sql('Full_Volunteer_dist', w, if_exists='replace', index=False)

#%%

query=''' SELECT *
FROM Full_Volunteer_dist
GROUP BY name_org, tax_yr
'''
df = pd.read_sql_query(query, w)
# df_fullbackup['formation_yr']=df_fullbackup['formation_yr'].fillna(2018).astype(int)
# df_fullbackup['formation_yr']=df_fullbackup['formation_yr'].astype(int)
# formation=df_fullbackup[df_fullbackup['formation_yr']<1318]

#%% Further data cleaning - start from here
df['org_form.association'] = df['org_form.association'].apply(lambda true: True if true else False)
df['org_form.corp'] = df['org_form.corp'].apply(lambda true: True if true else False)
df['org_form.other_form'] = df['org_form.other_form'].apply(lambda true: True if true else False)
df['org_form.trust'] = df['org_form.trust'].apply(lambda true: True if true else False)

# fill in formation yr with tax_yr if null of if more than 2018
df['formation_yr']=df['formation_yr'].fillna(value=df['tax_yr']).astype(int)
df['formation_yr']=df[['formation_yr','tax_yr']].apply(lambda x: x['tax_yr'] if x['formation_yr']>2018 else x['formation_yr'], axis=1).astype(int)
#%%
# changing this to formation duration; there are some anomolies with the formation duration, but we can leave for now since there are a small number of them
df['formation_yr']=df[['formation_yr','tax_yr']].apply(lambda x: 2018-x['formation_yr'] if x['formation_yr']!=0 else 2018-x['tax_yr'], axis=1) 
df=df.rename(columns={'formation_yr': 'formation_dur'})

#%% The formation yr of one organization, Accessory Council, was recorded to be 1194, which seemed to be an outlier. According to their website, they were founded on 1994, and this was corrected
# df['formation_yr']=df['formation_yr'].replace(max(df['formation_yr']), 24)
#%%
df_backup=df.copy()
df=df.drop(columns=['ntee_code_nccs_x', 'ntee_description_nccs_x'])
#%% Cleaning the dataset and fill in null values

# df=df_backup.copy()
# NY is 0 and NJ is 1
# df['state'] = df['state'].apply(lambda state: True if state=='NY' else False)
# df=df.rename(columns={'state': 'NJ'})
# df['discontinue_dispose25']=df_backup['discontinue_dispose25']
df['discontinue_dispose25']=df['discontinue_dispose25'].apply(lambda dd: 1 if dd=='true' else 0)
# df['volunteer_tot_ct']=df['volunteer_tot_ct'].fillna(0).astype(int)
df['material_diversion']=df['material_diversion'].apply(lambda md: 1 if md=='yes' else 0)
df['local_chapt']=df['local_chapt'].apply(lambda md: 1 if md=='yes' else 0)

# (df['material_diversion']=='no').sum()

#%%
ntee_encoded = pd.get_dummies(df['ntee_code_nccs_y'].str[0])
ntee_encoded.columns = ['code_' + col.upper() for col in ntee_encoded.columns]
df = pd.concat([df, ntee_encoded], axis=1)
df['code_Z']=df['ntee_code_nccs_y'].apply(lambda code: 1 if pd.isna(code)==True else 0)
df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'Of' else x)
df = df.drop(columns=['ntee_code_nccs_y', 'ntee_description_nccs_y'])
(df.iloc[:,-26:]==0).sum()

#%%
query=''' SELECT *
FROM Full_clean
'''
w= sa.create_engine(r'sqlite:///D:\Program_practices\Charity\open990database.sqlite')

df_backup= pd.read_sql_query(query, w)
#%%
df.to_sql('temp', w, if_exists='replace', index=False)

# # temp=df.loc[:,df['volunteer_tot_ct']<max(df['volunteer_tot_ct'])]
# # temp=df.loc[df['volunteer_tot_ct']!=df['volunteer_tot_ct'].max()]
# df['volunteer_tot_ct'].hist()

#%%
irrelevant= [ 'liability_tot_endyr', 'comp_currkeypersons_tot',
       'cash_noninterest_endyr', 'savingtempcash_endyr', 'pledges_net_endyr',
       'accountreceivable_net_endyr', 'invest_publicsec_endyr',
       'invest_othersec_endyr', 'invest_prog_endyr', 'asset_intangible_endyr',
       'asset_unrestrictnet_endyr', 'asset_temprestrictnet_endyr',
       'asset_permrestrictnet_endyr', 'liability_tot_beginyr','rev_tot_prioryr', 
       'expense_tot_prioryr', 'asset_net_beginyr', 'rev_less_expense_prioryr', 
       'asset_tot_beginyr', 'material_diversion','discontinue_dispose25']
df=df.drop(columns=irrelevant)

# drop independent voting member too (repetitive with the voting member)

non_num=['ein', 'name_org', 'tax_yr', 'city', 'state']
bool_var=['org_form.association', 'org_form.corp', 'org_form.other_form', 'org_form.trust',
          'local_chapt','code_A', 'code_B', 'code_C', 'code_D',
       'code_E', 'code_F', 'code_G', 'code_H', 'code_I', 'code_J', 'code_K',
       'code_L', 'code_M', 'code_N', 'code_O', 'code_P', 'code_Q', 'code_R',
       'code_S', 'code_T', 'code_U', 'code_V', 'code_W', 'code_X', 'code_Y',
       'code_Z']

df=df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
temp=df.copy()
df[bool_var]=df[bool_var].astype(bool)

df=df.apply(lambda x: x.astype(float) if (x.dtype=='int64') else x)
#%%
df=df.loc[(df['rev_less_expense_curyr'].quantile(.90)> df['rev_less_expense_curyr']) & (df['rev_less_expense_curyr'].quantile(.05)< df['rev_less_expense_curyr'])]

df=df.loc[(df['asset_net_endyr'].quantile(.90)> df['asset_net_endyr']) & (df['asset_net_endyr'].quantile(.05)< df['asset_net_endyr'])]

df=df.loc[(df['expense_tot_curyr'].quantile(.90)> df['expense_tot_curyr']) & (df['expense_tot_curyr'].quantile(.05)< df['expense_tot_curyr'])]

df=df.loc[(df['rev_tot_curyr'].quantile(.90)> df['rev_tot_curyr']) & (df['rev_tot_curyr'].quantile(.05)< df['rev_tot_curyr'])]

#%% 1500
df=df.loc[df['volunteer_tot_ct'].quantile(.99)> df['volunteer_tot_ct']]

#%%

# #%% Looks really skewed, so deleting the top 99% (3600.0 volunteers) or rev_less_expense_curyr: 1077362
# df=df.loc[(df['rev_less_expense_curyr'].quantile(.95)> df['rev_less_expense_curyr']) | (df['rev_less_expense_prioryr'].quantile(.95)> df['rev_less_expense_prioryr'])]
# df=df.loc[(df['rev_less_expense_curyr'].quantile(.05)< df['rev_less_expense_curyr']) | (df['rev_less_expense_prioryr'].quantile(.05)< df['rev_less_expense_prioryr'])]

# df=df.loc[(df['asset_net_beginyr'].quantile(.95)> df['asset_net_beginyr']) | (df['asset_net_endyr'].quantile(.95)> df['asset_net_endyr'])]
# df=df.loc[(df['asset_net_beginyr'].quantile(.05)< df['asset_net_beginyr']) | (df['asset_net_endyr'].quantile(.05)< df['asset_net_endyr'])]

# df=df.loc[(df['asset_tot_endyr'].quantile(.95)> df['asset_tot_endyr']) | (df['asset_tot_beginyr'].quantile(.95)> df['asset_tot_beginyr'])]
# df=df.loc[(df['asset_tot_endyr'].quantile(.05)< df['asset_tot_endyr']) | (df['asset_tot_beginyr'].quantile(.05)< df['asset_tot_beginyr'])]

# df=df.loc[(df['expense_tot_curyr'].quantile(.95)> df['expense_tot_curyr']) | (df['expense_tot_prioryr'].quantile(.95)> df['expense_tot_prioryr'])]
# df=df.loc[(df['expense_tot_curyr'].quantile(.05)< df['expense_tot_curyr']) | (df['expense_tot_prioryr'].quantile(.05)< df['expense_tot_prioryr'])]

# df=df.loc[(df['rev_tot_prioryr'].quantile(.95)> df['rev_tot_prioryr']) | (df['rev_tot_curyr'].quantile(.95)> df['rev_tot_curyr'])]
# df=df.loc[(df['rev_tot_prioryr'].quantile(.05)< df['rev_tot_prioryr']) | (df['rev_tot_curyr'].quantile(.05)< df['rev_tot_curyr'])]

# df=df.loc[df['volunteer_tot_ct'].quantile(.99)> df['volunteer_tot_ct']]

# # df=df.loc[df['volunteer_tot_ct'].quantile(.99)> df['volunteer_tot_ct']]
#%%

# df=df.loc[df['rev_less_expense_curyr'].quantile(.95)> df['rev_less_expense_curyr']]
# df=df.loc[df['rev_less_expense_curyr'].quantile(.05)< df['rev_less_expense_curyr']]

# df=df.loc[df['rev_less_expense_prioryr'].quantile(.95)> df['rev_less_expense_prioryr']]
# df=df.loc[df['rev_less_expense_prioryr'].quantile(.05)< df['rev_less_expense_prioryr']]

# df=df.loc[df['asset_net_beginyr'].quantile(.95)> df['asset_net_beginyr']]
# df=df.loc[df['asset_net_beginyr'].quantile(.05)< df['asset_net_beginyr']]

# df=df.loc[df['asset_net_endyr'].quantile(.95)> df['asset_net_endyr']]
# df=df.loc[df['asset_net_endyr'].quantile(.05)< df['asset_net_endyr']]

# df=df.loc[df['asset_tot_endyr'].quantile(.95)> df['asset_tot_endyr']]
# df=df.loc[df['asset_tot_endyr'].quantile(.05)< df['asset_tot_endyr']]

# df=df.loc[df['asset_tot_beginyr'].quantile(.95)> df['asset_tot_beginyr']]
# df=df.loc[df['asset_tot_beginyr'].quantile(.05)< df['asset_tot_beginyr']]

# df=df.loc[df['expense_tot_curyr'].quantile(.95)> df['expense_tot_curyr']]
# df=df.loc[df['expense_tot_curyr'].quantile(.05)< df['expense_tot_curyr']]

# df=df.loc[df['rev_tot_prioryr'].quantile(.95)> df['rev_tot_prioryr']]
# df=df.loc[df['rev_tot_prioryr'].quantile(.05)< df['rev_tot_prioryr']]

# df=df.loc[df['rev_tot_curyr'].quantile(.95)> df['rev_tot_curyr']]
# df=df.loc[df['rev_tot_curyr'].quantile(.05)< df['rev_tot_curyr']]

# df=df.loc[df['volunteer_tot_ct'].quantile(.99)> df['volunteer_tot_ct']]
#%%

# df['volunteer_tot_ct'].describe()

#%% Moving to data exploration... we can definitely delete some features to only have a net change 
# but some organizations have negative revenue to do expenses. Let's leave it for now
from sklearn.model_selection import StratifiedShuffleSplit

# Divide them into given bins to ensure that the number of ratings distribution represents the train and test sets
bins = [0, 10, 50, 100, 5000, np.inf]
labels=[1, 2, 3, 4, 5]
# create a new column containing the ratings bin the game is associated with
df['volunteer_tot_ct_cut'] = pd.cut(df['volunteer_tot_ct'], bins=bins, labels=labels)

# split data into training (80%) and testing (20%) sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_ind, test_ind in split.split(df, df['volunteer_tot_ct_cut']):
    train_set, test_set = df.iloc[train_ind], df.iloc[test_ind]

test_label = test_set['volunteer_tot_ct'].copy()
test_identity = test_set[['name_org', 'ein']].copy()
test_set = test_set.drop(columns=['volunteer_tot_ct_cut','volunteer_tot_ct','name_org','tax_yr', 'ein', 'city', 'state'])

train_label = train_set['volunteer_tot_ct'].copy()
train_identity = train_set[['name_org', 'ein']].copy()
train_set_binned = train_set['volunteer_tot_ct_cut'].copy() # label identifying the ratings bin the game occupies
train_set = train_set.drop(columns=['volunteer_tot_ct_cut','volunteer_tot_ct','name_org','tax_yr', 'ein', 'city', 'state']) 
test_set.info()

#%%
train_set.to_sql('train_set', w, if_exists='replace', index=False)
train_label.to_sql('train_label', w, if_exists='replace', index=False)
test_set.to_sql('test_set', w, if_exists='replace', index=False)
test_label.to_sql('test_label', w, if_exists='replace', index=False)

#%%
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew
from scipy.stats import boxcox

class SkewCorrector(BaseEstimator, TransformerMixin):
    
    def __init__(self, skew_bound=0.2): # skew_bound is amount of skew that is acceptable
        self.skew_bound = skew_bound
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_val = X.to_numpy()
        # different distributions require different transforms; indicies are defined below for specific data types
        ind_skewed = np.absolute(skew(X_val)) > self.skew_bound
        ind_right_skew = skew(X_val) > 0
        ind_left_skew = skew(X_val) < 0
        # ind_no_zeros = [0 not in X_val[:,col] for col in np.arange(X_val.shape[1])]
        ind_has_zeros = [0 in X_val[:,col] for col in np.arange(X_val.shape[1])]
        # if all elements in a feature column are positive, the sum of all those should be equal to the number of rows in that feature matrix
        # ind_positive = [np.sign(X_val[:,col]).sum() == X_val.shape[0] for col in np.arange(X_val.shape[1])]

        # # transform right and left skewed data that does not include zero values with boxcox
        # X_trans, _ = np.apply_along_axis(boxcox, 0, X_val[:,ind_skewed & ind_no_zeros & ind_positive]) # returns list of arrays
        # X_val[:,ind_skewed & ind_no_zeros & ind_positive] = np.vstack(X_trans).T
        
        # # transform right skewed data that contains zero values with log plus one
        X_trans = np.log1p(X_val[:,ind_skewed & ind_right_skew & ind_has_zeros])
        X_val[:,ind_skewed * ind_right_skew * ind_has_zeros] = X_trans

        
        # # transform left skewed data that contains zero values by increasing power incrementally
        for pwr in range(2, 5): #arbitarily set the power limit to 5; maybe make this a user-defined parameter?
            X_trans = X_val[:,ind_skewed * ind_left_skew * ind_has_zeros]**pwr
            # if skew(X_trans) < self.skew_bound:
            #     break
        
        X_val[:,ind_skewed * ind_left_skew * ind_has_zeros] = X_trans
        
        # Update the input data frame with transformed values
        X_out = pd.DataFrame(X_val, index=X.index, columns=X.columns)
        return X_out
    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    ('trans_skew', SkewCorrector(skew_bound=0.2)),
    ('std_scaler', StandardScaler())
])
numeric_feat = train_set.dtypes[train_set.dtypes == 'float64'].index
cat_feat = train_set.dtypes[train_set.dtypes != 'float64'].index

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numeric_feat)
])

train_set_processed = train_set.copy()
# Correct negative values by adding constants
# minimum_val=df[numeric_feat].min()
  
# temp=df.loc[:,df['volunteer_tot_ct']<max(df['volunteer_tot_ct'])]
for feat_idx, feat in enumerate(numeric_feat):
    if (train_set_processed[feat]<0).any()==True:
        train_set_processed[feat]= train_set_processed[feat]+ abs(min(train_set_processed[feat]))

# min_train_processed=train_set_processed[numeric_feat].min()
# train_set_processed[numeric_feat].hist(bins=10,figsize=(20,20))
# train_set_processed=train_set_processed.fillna(train_set_processed.mean(), inplace=True)
# train_set_processed.info()

train_set_processed[numeric_feat] = full_pipeline.fit_transform(train_set_processed[numeric_feat])

train_set[numeric_feat].hist(bins=10, figsize=(20,20))
train_set_processed[numeric_feat].hist(bins=10, figsize=(20,20))
#%%
from sklearn.model_selection import cross_val_score, StratifiedKFold

def init_cv_gen():
    '''Helper function that returns a generator for cross-validation'''
    skf = StratifiedKFold(5, shuffle=True, random_state=42).split(train_set_processed, train_set_binned)
    return skf

def cv_score(estimator):
    '''Generates cross validation scores; used to keep code clean when testing different models (estimator)'''
    cv = init_cv_gen()
    cv_score = cross_val_score(estimator=estimator, X=train_set_processed, y=train_label, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    return cv_score

def show_cv_result(scores):
    '''Print mean and standard deviation of an array of cross validation scores'''
    print(f'Cross validation mean score is : {np.mean(scores)} ± {np.std(scores)}')

#%%
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(train_set_processed, train_label)

scores = cv_score(reg)
show_cv_result(-scores)
    
# Cross validation mean score is : 102.63889869368822 ± 0.7685925246738338
# 79.0543953032712 ± 0.7871683782583261
#%%
from sklearn.linear_model import ElasticNet

ENreg = ElasticNet()
ENreg.fit(train_set_processed, train_label)

ENscores = cv_score(ENreg)
show_cv_result(-ENscores)

#Cross validation mean score is : 103.69660330724726 ± 0.7343927227462941
#80.48191721097434 ± 0.7210385489374425
#%%
from sklearn.linear_model import SGDRegressor

SGDreg = SGDRegressor(loss='huber',max_iter=5000, tol=1e-3)
SGDreg.fit(train_set_processed, train_label)

SGDscores = cv_score(SGDreg)
show_cv_result(-SGDscores)
#Cross validation mean score is : 83.46956090926645 ± 0.96953112044042894
# 65.85259779194307 ± 0.8383987379735028
#%%
from sklearn.svm import LinearSVR

SVRreg = LinearSVR()
SVRreg.fit(train_set_processed, train_label)

SVRscores = cv_score(SVRreg)
show_cv_result(-SVRscores)
#Cross validation mean score is : 82.61712906240216 ± 0.968301494067034
#64.99349558052299 ± 0.819643910375903
#%%
from sklearn.neighbors import KNeighborsRegressor

KNreg = KNeighborsRegressor()
KNreg.fit(train_set_processed, train_label)

KNscores = cv_score(KNreg)
show_cv_result(-KNscores)

#Cross validation mean score is : 105.54859942524288 ± 0.5803541664895314
#83.65854769695443 ± 0.6234783163982724
#%%
from sklearn.ensemble import RandomForestRegressor

RFRreg = RandomForestRegressor(n_estimators=100, random_state=42)
RFRreg.fit(train_set, train_label)

RFRscores = cv_score(RFRreg)
show_cv_result(-RFRscores)
# Cross validation mean score is : 105.4876955515767 ± 0.8889279716675633
# 82.0435323433174 ± 0.6593399599068464
#%%
from sklearn.ensemble import GradientBoostingRegressor

GBRreg = GradientBoostingRegressor(loss='huber', random_state=42)
GBRreg.fit(train_set, train_label)

GBRscores = cv_score(GBRreg)
show_cv_result(-GBRscores)
#Cross validation mean score is : 85.55880096145253 ± 0.9566986167681454
# Cross validation mean score is : 68.25811253664408 ± 0.7472996928973249
#%% LinearSVR SGDregressor, GBR

from sklearn.model_selection import GridSearchCV
# SGD
param_grid = [
    {
        'penalty': ['none', 'elasticnet'],
        'loss': ['huber', 'epsilon_insensitive'],
        'alpha': [1, 0.1, 1e-2],
        'learning_rate': ['invscaling', 'adaptive']
    }
]
SGDreg = SGDRegressor(max_iter=2000, tol=1e-3, early_stopping=True)

sgd_grid = GridSearchCV(SGDreg, param_grid, cv=5, scoring='neg_mean_absolute_error', return_train_score=True, n_jobs=-1)
sgd_grid.fit(train_set_processed, train_label)
-sgd_grid.best_score_
#65.01403572101756
''' SGDRegressor(alpha=1, average=False, early_stopping=True, epsilon=0.1,
             eta0=0.01, fit_intercept=True, l1_ratio=0.15,
             learning_rate='adaptive', loss='epsilon_insensitive',
             max_iter=2000, n_iter_no_change=5, penalty='none', power_t=0.25,
             random_state=None, shuffle=True, tol=0.001,
             validation_fraction=0.1, verbose=0, warm_start=False)
'''
#%%

# GBR
param_grid = [
    {
        'max_depth': [9, 15, 20],
        'learning_rate': [0.1],
        'loss' : ['huber', 'lad'],
        'warm_start' :[True],
        'min_samples_split' : [100, 120]
    }
]

GBRreg = GradientBoostingRegressor(random_state=42)

GBR_grid = GridSearchCV(GBRreg, param_grid, cv=5, scoring='neg_mean_absolute_error', return_train_score=True, n_jobs=-1)
GBR_grid.fit(train_set, train_label)
-GBR_grid.best_score_
GBR_grid.best_estimator_
#63.328281547064385
'''
GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='lad',
                          max_depth=15, max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=120,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=42, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=True)
'''
# 79.59250773212361
#%%
# SVR
param_grid = [
    {
        'epsilon': [0, .1, 1, 10, 100],
        'C': [1, 10, 100, 1000]
    }
]

SVRreg = LinearSVR(max_iter=10000)

SVR_grid = GridSearchCV(SVRreg, param_grid, cv=5, scoring='neg_mean_absolute_error', return_train_score=True, n_jobs=-1)
SVR_grid.fit(train_set_processed, train_label)
-SVR_grid.best_score_
SVR_grid.best_estimator_
''' LinearSVR(C=10, dual=True, epsilon=1, fit_intercept=True, intercept_scaling=1.0,
          loss='epsilon_insensitive', max_iter=10000, random_state=None,
          tol=0.0001, verbose=0)
'''
#64.98908264223675
#%%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
train_predict = GBR_grid.best_estimator_.predict(train_set).round(0)
test_predict = GBR_grid.best_estimator_.predict(test_set).round(0)

# mean absolute error calculation
mae_train = mean_absolute_error(train_label, train_predict)
mae_test = mean_absolute_error(test_label, test_predict)
# median absolute error calculation
medae_train = median_absolute_error(train_label, train_predict)
medae_test = median_absolute_error(test_label, test_predict)
# print results
print(f'Training set mean absolute error and median absolute error is {mae_train} and {medae_train}, respectively')
print(f'Test set mean absolute error and median absolute error is {mae_test} and {medae_test}, respectively')
#Training set mean absolute error and median absolute error is 57.56726654920715 and 14.0, respectively
#Test set mean absolute error and median absolute error is 62.484949159367766 and 20.0, respectively
#%%
from sklearn.inspection import permutation_importance

result = permutation_importance(GBR_grid.best_estimator_, test_set, test_label, n_repeats=10,
                                random_state=42)

# sort to plot in order of importance
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(16,16))

ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=test_set.columns[sorted_idx])

ax.set_title("Permutation Importances (based on test set)")
#%%
ex = df[(df['city']=='Hoboken') & (df['state']=='NJ')]

ex_summary = ex[['name_org','volunteer_tot_ct']].copy()
ex_feat = ex[['org_form.association', 'org_form.corp', 'org_form.other_form',
       'org_form.trust', 'formation_dur', 'gross_receipts', 'voting_memb_ct',
       'voting_indepmemb_ct', 'employee_tot_ct', 'rev_tot_curyr',
       'expense_tot_curyr', 'rev_less_expense_curyr', 'asset_tot_endyr',
       'asset_net_endyr', 'local_chapt', 'code_A', 'code_B', 'code_C',
       'code_D', 'code_E', 'code_F', 'code_G', 'code_H', 'code_I', 'code_J',
       'code_K', 'code_L', 'code_M', 'code_N', 'code_O', 'code_P', 'code_Q',
       'code_R', 'code_S', 'code_T', 'code_U', 'code_V', 'code_W', 'code_X',
       'code_Y', 'code_Z']]
ex_summary['predicted_voluntter'] = GBR_grid.best_estimator_.predict(ex_feat).round(0)

ex_summary

#%%
ex2= df.sample(n = 15)

ex_summary2 = ex2[['name_org','volunteer_tot_ct']].copy()
ex_feat2 = ex2[['org_form.association', 'org_form.corp', 'org_form.other_form',
       'org_form.trust', 'formation_dur', 'gross_receipts', 'voting_memb_ct',
       'voting_indepmemb_ct', 'employee_tot_ct', 'rev_tot_curyr',
       'expense_tot_curyr', 'rev_less_expense_curyr', 'asset_tot_endyr',
       'asset_net_endyr', 'local_chapt', 'code_A', 'code_B', 'code_C',
       'code_D', 'code_E', 'code_F', 'code_G', 'code_H', 'code_I', 'code_J',
       'code_K', 'code_L', 'code_M', 'code_N', 'code_O', 'code_P', 'code_Q',
       'code_R', 'code_S', 'code_T', 'code_U', 'code_V', 'code_W', 'code_X',
       'code_Y', 'code_Z']]
ex_summary2['predicted_voluntter'] = GBR_grid.best_estimator_.predict(ex_feat2).round(0)

ex_summary2

#%% Too volatile due to the skewness; only put through data that are extremely skewed as 0.99 and 0.01

fig, axs = plt.subplots(2,5, figsize=(20, 15))
axs = axs.flatten()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical features to test
star_feats = ['formation_dur', 'gross_receipts', 'voting_memb_ct',
       'voting_indepmemb_ct', 'employee_tot_ct', 'rev_tot_curyr',
       'expense_tot_curyr', 'rev_less_expense_curyr', 'asset_tot_endyr',
       'asset_net_endyr']

# features of float type (rest are integers)
# float_feats = ['formation_dur', 'gross_receipts', 'voting_memb_ct',
#        'voting_indepmemb_ct', 'employee_tot_ct', 'rev_tot_curyr',
#        'expense_tot_curyr', 'rev_less_expense_curyr', 'asset_tot_endyr',
#        'asset_net_endyr']

# xlabels to add based on feature being plotted
feats_xlabel = ['duration of establishment', 'gross receipts','# of voting members', 
                '# of independent voting member', '# of total employee', 'total revenue',
                'total expense', 'revenue - expense', 'total asset', 
                'asset-liability'
               ]
non_trainable_feat = ['name_org', 'ein', 'volunteer_tot_ct', 'volunteer_tot_ct_cut', 
                      'city', 'state', 'tax_yr']

# loop through each feature and produce a subplot
for feat_ind, feat in enumerate(star_feats):
    
    # isolate predictor variables to input in model
    ex_feat = ex.drop(columns=non_trainable_feat)
    
    # if feature is float type, test a range of values that lie in between the 15th and 85th percentiles 
    # (in terms of frequency of occurance); create if statements like shown below for price

    low_bnd = df[[feat]].quantile(q=0.01, axis=0).item()
    up_bnd = df[[feat]].quantile(q=0.99, axis=0).item()
    
    # price is heavily skewed; if we consider only titles that lie in the 15-85th percentile range, they would all have a price of $0.00
    # consequently, we will consider all but the most extreme outliers in price
    # if feat == 'price':
    #     low_bnd = df[[feat]].quantile(q=0.01, axis=0).item()
    #     up_bnd = df[[feat]].quantile(q=0.99, axis=0).item()
    
    # for float types, we plot equispaced points between the percentile range described above
    x_rng = np.linspace(low_bnd, up_bnd, num=10)

# if feature is integer type, likewise test values between the 15th and 85th percentile
# difference from float is that we test all integer values within the prescribed range

    # preallocate variable that will contain predicted number of ratings 
    predictions = np.zeros((len(x_rng), len(ex_feat)))

    # calculate predictions for example games
    for ind, val in enumerate(x_rng):
        ex_feat[feat] = val
        predictions[ind,:] = GBR_grid.best_estimator_.predict(ex_feat).round(0)
    
    # plotting code
    fig.tight_layout()
    axs[feat_ind].plot(x_rng, predictions, linestyle='--', marker='o')
    axs[feat_ind].set_title(feat)
    axs[feat_ind].set_ylabel('predicted volunteer')
    axs[feat_ind].set_xlabel(feats_xlabel[feat_ind])
    
fig.legend(ex_summary['name_org'].tolist(), 
           loc='upper center', 
           ncol=3,
           bbox_to_anchor=(0.5, 1.03))

#%%
import seaborn as sns

# isolate predictor variables to input in model
bar_data = ex.drop(columns=non_trainable_feat)

# categorical features to test
cat_feat = ['org_form.association', 'org_form.corp', 'org_form.other_form', 'org_form.trust',
          'local_chapt','code_A', 'code_B', 'code_C', 'code_D',
       'code_E', 'code_F', 'code_G', 'code_H', 'code_I', 'code_J', 'code_K',
       'code_L', 'code_M', 'code_N', 'code_O', 'code_P', 'code_Q', 'code_R',
       'code_S', 'code_T', 'code_U', 'code_V', 'code_W', 'code_X', 'code_Y',
       'code_Z']

# titles for subplots
bar_title = ['Association', 'Corporation', 'Others', 'Trust', 
             'Local Chapter', 'IRS Activity Code A', 'code_B', 'code_C', 'code_D',
       'code_E', 'code_F', 'code_G', 'code_H', 'code_I', 'code_J', 'code_K',
       'code_L', 'code_M', 'code_N', 'code_O', 'code_P', 'code_Q', 'code_R',
       'code_S', 'code_T', 'code_U', 'code_V', 'code_W', 'code_X', 'code_Y',
       'Unknown']
# loop through each feature and generate a subplot
for feat, title in zip(cat_feat, bar_title):
    
    # preallocate DataFrame containing predictions
    bar_data_pred = pd.DataFrame()
    
    # set tested feature to false for all example games
    bar_data_false = bar_data.copy()
    bar_data_false[feat] = False
    
    # set tested feature to true for all example games
    bar_data_true = bar_data.copy()
    bar_data_true[feat] = True
    
    # combine 'false' and 'true' versions
    bar_data_pred = bar_data_true.append(bar_data_false)
    
    # almost all games with in-app purchases are free
    # for this specific feature we also change the price to 0.00

    # calculate predictions for example games
    bar_data_pred['predicted_volunteer'] = GBR_grid.best_estimator_.predict(bar_data_pred)
    
    # add game titles to DataFrame (in proper order)
    # note that the list of titles has to be repeated
    # one set for feature set to True, another for the feature set to False
    bar_data_pred['name_org'] = 2 * ex_summary['name_org'].tolist()

    # plotting code
    g = sns.catplot(x='name_org', y='predicted_volunteer', hue=feat, 
            data=bar_data_pred, kind='bar', height=4)
    
    plt.title(title)
    g.set_xticklabels(rotation=90)
#%%
df.to_sql('temp', w, if_exists='replace', index=False)

#%%
numeric_feat = train_set.dtypes[(train_set.dtypes == 'float64') | (train_set.dtypes == 'int64')].index
train_set[numeric_feat].hist(bins=10, figsize=(20,20))
#%%
df=df[['name_org','tax_yr','phone','address','city','state', 'ntee_description_nccs_y', 'employee_tot_ct', 'volunteer_tot_ct', 'rev_tot_prioryr', 'rev_tot_curyr' ]]
df.to_sql('temp2', w, if_exists='replace', index=False)
#%% We only ended up with 5000 data. I will export this for Frank but I will not use it. Moving to exploit the whole dataset after



