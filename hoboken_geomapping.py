# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:03:23 2020

@author: Andrea
"""


#%% geocoding for Charity Quest
from geopy.geocoders import Nominatim
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
import numpy as np
from geopy.extra.rate_limiter import RateLimiter
#%%
query=''' SELECT *
FROM hoboken_phone_address'''

e = sa.create_engine(r'sqlite:///D:\Program_practices\Charity\open990database.db')
df= pd.read_sql_query(query, e)
#%%
query=''' SELECT *
FROM hoboken_merger'''

e = sa.create_engine(r'sqlite:///D:\Program_practices\Charity\open990database.db')
df_merg= pd.read_sql_query(query, e)
key=df_merg[['ein', 'ntee_code_nccs']]
#%%
df=pd.merge(df,key, on='ein', how='left')
df = df.drop_duplicates(subset='ein', keep="last")
df['ntee_code_nccs_x']=df['ntee_code_nccs_x'].fillna(df['ntee_code_nccs_y'])
df=df.drop(columns='ntee_code_nccs_y')

#%%
locator = Nominatim(user_agent='myGeocoder')
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
df['address']=df['address'] +', Hoboken, NJ'
df['location'] = df['address'].apply(geocode)
#%%
df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
df['latitude']=df['point'].str[0]
df['longitude']=df['point'].str[1]
df_backup=df.copy()

#%%
df['ntee_code_nccs_x']=df['ntee_code_nccs_x'].str[0]
df['ntee_code_nccs_x']=df['ntee_code_nccs_x'].apply(lambda code: 'Z' if pd.isna(code)==True else code)
uniq_code=list(set(df['ntee_code_nccs_x']))
#%%
# ntee_encoded = pd.get_dummies(df['ntee_code_nccs_x'].str[0])
# ntee_encoded.columns = ['code_' + col.upper() for col in ntee_encoded.columns]
# ntee_encoded=ntee_encoded.astype(bool)
# df = pd.concat([df, ntee_encoded], axis=1)
# df['code_Z']=df['ntee_code_nccs_x'].apply(lambda code: 1 if pd.isna(code)==True else 0)
# df['code_Z']=df['code_Z'].astype(bool)
#%%

colors=['darkred', 'blue', 'green', 'purple', 'orange','lightgray', 'gray',
        'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue','black']

def codecolors(counter):
    for i in range(0,len(colors)):
        if counter == uniq_code[i]:
            return colors[i]
    
# df['colors'] = np.where(df['ntee_code_nccs_x']==uniq_code[0], 'red')

df['color'] = df['ntee_code_nccs_x'].apply(codecolors)

#%%
import folium
# df['ntee_code_nccs_x'] = df['ntee_code_nccs_x'].str[0]
# df['ntee_code_nccs_x']=df['ntee_code_nccs_x'].fillna(df['ntee_code_nccs_y'])

df['longitude']=df['longitude'].fillna(-74.032694)
df['latitude']=df['latitude'].fillna(40.744070)
m = folium.Map(location=[40.744070, -74.032694],tiles='CartoDB positron', zoom_start=15)

ntee_label=['Voluntary Health Associations & Medical Disciplines','Religion-Related',
            'Unknown/not specified', 'Housing & Shelter', 'Health Care',
            'Human Services', 'Civil Rights, Social Action & Advocacy',
            'Mutual & Membership Benefit', 'Recreation & Sports', 'Employment',
            'Arts, Culture & Humanities','Community Improvement & Capacity Building',
            'Education', 'Philanthropy, Voluntarism & Grantmaking Foundations']
def codecolors(counter):
    for i in range(0,len(ntee_label)):
        if counter == uniq_code[i]:
            return ntee_label[i]
df['ntee_label'] = df['ntee_code_nccs_x'].apply(codecolors)  
df['name_org']=df['name_org'].str.title() 
df['address']=df['address'].str.title()
df['address']=df['address'].str[:-13]
df['phone']=df['phone'].apply(lambda x: 'N/A' if pd.isna(x)==True else x)
#%%
#https://georgetsilva.github.io/posts/mapping-points-with-folium/
from folium.plugins import MarkerCluster

marker_cluster = MarkerCluster().add_to(m)
df.apply(lambda row:folium.Marker(
    location=[row['latitude'], row['longitude']], 
    popup=row['name_org']+'  '+row['address'] +'  '+row['phone'], max_width=500,min_width=400,
    icon=folium.Icon(color=row['color'], icon='hand-holding-heart')).add_to(marker_cluster),
         axis=1)
legend_html =   '''
                <div style="position: fixed; 
                            bottom: 40px; left: 40px; width: 410px; height:380px; 
                            border:2px solid grey; z-index:9999; font-size:12px;
                            ">&nbsp; Ntee code <br>
                              &nbsp; Voluntary Health Associations & Medical Disciplines &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkred"></i><br>
                              &nbsp; Religion-Related &nbsp; <i class="fa fa-map-marker fa-2x" style="color:dodgerblue"></i><br>
                              &nbsp; Unknown/not specified &nbsp; <i class="fa fa-map-marker fa-2x" style="color:forestgreen"></i><br>
                              &nbsp; Housing & Shelter &nbsp; <i class="fa fa-map-marker fa-2x" style="color:mediumorchid"></i><br>
                              &nbsp; Health Care &nbsp; <i class="fa fa-map-marker fa-2x" style="color:orange"></i><br>
                              &nbsp; Human Services &nbsp; <i class="fa fa-map-marker fa-2x" style="color:gray"></i><br>
                              &nbsp; Civil Rights, Social Action & Advocacy &nbsp; <i class="fa fa-map-marker fa-2x" style="color:dimgray"></i><br>
                              &nbsp; Mutual & Membership Benefit &nbsp; <i class="fa fa-map-marker fa-2x" style="color:steelblue"></i><br>
                              &nbsp; Recreation & Sports &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkolivegreen"></i><br>
                              &nbsp; Employment &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkcyan"></i><br>
                              &nbsp; Arts, Culture & Humanities &nbsp; <i class="fa fa-map-marker fa-2x" style="color:indigo"></i><br>
                              &nbsp; Community Improvement & Capacity Building &nbsp; <i class="fa fa-map-marker fa-2x" style="color:magenta"></i><br>
                              &nbsp; Education &nbsp; <i class="fa fa-map-marker fa-2x" style="color:lightskyblue"></i><br>
                              &nbsp; Philanthropy, Voluntarism & Grantmaking Foundations &nbsp; <i class="fa fa-map-marker fa-2x" style="color:black"></i>
                </div>
                ''' 

m.get_root().html.add_child(folium.Element(legend_html))

m.save('hoboken_map.html')
