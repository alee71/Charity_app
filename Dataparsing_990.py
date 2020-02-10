# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:15:54 2020

@author: Andrea
"""
import pandas as pd 
import sqlalchemy as sa
from pathlib import Path
from sqlalchemy import create_engine
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Point
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
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
FROM Full_data
WHERE upper(state)='NJ'
GROUP BY name_org, tax_yr
'''
nj = pd.read_sql_query(query, w)
nj.to_sql('NJ', w, if_exists='replace', index=False)

query=''' SELECT *
FROM NJ
WHERE lower(city)='hoboken'
GROUP BY name_org, tax_yr
'''
hoboken = pd.read_sql_query(query, w)
hoboken.to_sql('Hoboken', w, if_exists='replace', index=False)

query=''' SELECT *
FROM Full_data
WHERE upper(state)='NY'
GROUP BY name_org, tax_yr
'''
ny = pd.read_sql_query(query, w)
ny.to_sql('NY', w, if_exists='replace', index=False)

#%%
query=''' SELECT *
FROM NJ
WHERE lower(city)='hoboken' OR lower(city)='jersey city' OR lower(city)='palisades park' OR lower(city)='fort lee'
OR lower(city)='weehawken' OR lower(city)='union city' OR lower(city)='west new york'
OR lower(city)='north bergen' OR lower(city)='west new york' OR lower(city)='guttenberg'
OR lower(city)='fairview' OR lower(city)='cliffside park' OR lower(city)='edgewater' 
OR lower(city)='ridgefield' OR lower(city)='bayonne' OR lower(city)='secaucus' OR lower(city)='leonia' 
GROUP BY name_org, tax_yr
'''
nj_area = pd.read_sql_query(query, w)
nj_area.to_sql('NJ_area', w, if_exists='replace', index=False)

#%%
c_hoboken= hoboken[['name_org', 'tax_yr', 'phone', 'address', 'city', 'formation_yr']]
c_hoboken.to_sql('Hoboken', w, if_exists='replace', index=False)
#%%
query=''' SELECT y1.*
FROM NJ_area y1
WHERE y1.tax_yr = (SELECT max(y2.tax_yr)
                   FROM NJ_area y2
                   WHERE y2.ein=y1.ein)
'''
nj_area_tax = pd.read_sql_query(query, w)
#%%
query=''' SELECT *
FROM NY
WHERE lower(city)='brooklyn' OR lower(city)='new york' OR lower(city)='bronx' OR lower(city)='flushing' OR lower(city)='staten island' OR 
lower(city)='manhatthan' OR lower(city)='long island city' OR lower(city)='new york city' OR lower(city)='astoria' OR lower(city)='ny' OR lower(city)='new york ny' OR lower(city)='queens'
'''
ny_area = pd.read_sql_query(query, w)
ny_area.to_sql('NYC_area', w, if_exists='replace', index=False)

query=''' SELECT y1.*
FROM NYC_area y1
WHERE y1.tax_yr = (SELECT max(y2.tax_yr)
                   FROM NYC_area y2
                   WHERE y2.ein=y1.ein)
'''
ny_area_tax = pd.read_sql_query(query, w)

df=pd.concat([nj_area_tax, ny_area_tax]).reset_index(drop=True).drop_duplicates('ein')
df_ext=pd.concat([nj_area, ny_area]) 
df.to_sql('NJNYarea_dist', w, if_exists='replace', index=False)
#%% Loading previous database - ignore for notebook
w= sa.create_engine(r'sqlite:///D:\Program_practices\Charity\open990database.sqlite')

query=''' SELECT *
FROM NJNYarea
'''
df = pd.read_sql_query(query, w)
#%%
key=df_ext[['ein', 'ntee_code_nccs', 'ntee_description_nccs']]
key=key.drop_duplicates().dropna().reset_index(drop=True)
df=pd.merge(df,key, on='ein', how='left')
df=df.drop_duplicates('ein')

#%% Further data cleaning - start from here
irrelevant=['tax_date_begin', 'tax_date_end', 'doing_business_as', 'phone', 'website', 
             'address', 'city', 'zip']
df=df.drop(columns=irrelevant)

df['org_form.association'] = df['org_form.association'].apply(lambda true: True if true else False)
df['org_form.corp'] = df['org_form.corp'].apply(lambda true: True if true else False)
df['org_form.other_form'] = df['org_form.other_form'].apply(lambda true: True if true else False)
df['org_form.trust'] = df['org_form.trust'].apply(lambda true: True if true else False)

df['formation_yr']=df['formation_yr'].fillna(0).astype(int)
df['formation_yr']=df[['formation_yr','tax_yr']].apply(lambda x: 2018-x['formation_yr'] if x['formation_yr']!=0 else 2018-x['tax_yr'], axis=1) 
# The formation yr of one organization, Accessory Council, was recorded to be 1194, which seemed to be an outlier. According to their website, they were founded on 1994, and this was corrected
df['formation_yr']=df['formation_yr'].replace(max(df['formation_yr']), 24)
#%%
df_backup=df.copy()
df=df.drop(columns=['ntee_code_nccs_x', 'ntee_description_nccs_x'])
#%%

df=df.rename(columns={'formation_yr': 'formation_dur'})
# NY is 0 and NJ is 1
df['state'] = df['state'].apply(lambda state: True if state=='NY' else False)
df=df.rename(columns={'state': 'NJ'})
# df['discontinue_dispose25']=df_backup['discontinue_dispose25']
df['discontinue_dispose25']=df['discontinue_dispose25'].apply(lambda dd: 1 if dd=='true' else 0)
# df['volunteer_tot_ct']=df['volunteer_tot_ct'].fillna(0).astype(int)
df['material_diversion']=df['material_diversion'].apply(lambda md: 1 if md=='yes' else 0)
df['local_chapt']=df['local_chapt'].apply(lambda md: 1 if md=='yes' else 0)
df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'Of' else x)
# (df['material_diversion']=='no').sum()

#%%
ntee_encoded = pd.get_dummies(df['ntee_code_nccs_y'].str[0])
ntee_encoded.columns = ['code_' + col.upper() for col in ntee_encoded.columns]
df = pd.concat([df, ntee_encoded], axis=1)
df['code_Z']=df['ntee_code_nccs_y'].apply(lambda code: 1 if pd.isna(code)==True else 0)
df = df.drop(columns=['ntee_code_nccs_y', 'ntee_description_nccs_y'])
(df.iloc[:,-26:]==0).sum()
#%%
# Change data type except for the name of the organization
df.loc[:,df.columns!='name_org']= df.loc[:,df.columns!='name_org'].apply(pd.to_numeric) 

# df_backup2=df.copy()

#%%
# query=''' SELECT *
# FROM temp_code
# '''
# df = pd.read_sql_query(query, w)
#%%
df.to_sql('NJNY_clean', w, if_exists='replace', index=False)

temp=df.loc[:,df['volunteer_tot_ct']<max(df['volunteer_tot_ct'])]
temp=df.loc[df['volunteer_tot_ct']!=df['volunteer_tot_ct'].max()]
temp['volunteer_tot_ct'].hist()

#%%
import seaborn as sn
df=df[df['volunteer_tot_ct'] !=0]

corrMatrix = df_vot.corr()
sn.heatmap(corrMatrix, annot=True)
#%%
df=df[['name_org','tax_yr','phone','address','city','state', 'ntee_description_nccs_y', 'employee_tot_ct', 'volunteer_tot_ct', 'rev_tot_prioryr', 'rev_tot_curyr' ]]
df.to_sql('NJNYarea_Frank', w, if_exists='replace', index=False)
#%% We only ended up with 5000 data. I will export this for Frank but I will not use it. Moving to exploit the whole dataset after


#%% Moving to data exploration... we can definitely delete some features to only have a net change 
# but some organizations have negative revenue to do expenses. Let's leave it for now


