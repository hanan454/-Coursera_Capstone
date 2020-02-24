#!/usr/bin/env python
# coding: utf-8

# <h1 align=center> Exploring top venues around NYC's universities </h1>

# ### Business Problem

# The idea from this project is to help who want to gain benefits from the trending venues around to the universities of the New York City such as student's services stores owners who want to open their stores near to places visible and known by providing a clear understanding of the trending places around the top universities and academies in New York City and clustering them based on their common characteristics and that will help them in making their decision.

# ### Interest

# The targeted audience of this project are those who want to make a business targeting the students, such as students' services stores, coffee shops dedicated to the study and so on, and want to open their business in places known and visible by the students and obtain a competitive advantage.

# ### Data acquisition

# <ul>
#  <li> Firstly, I use data of the universities and academies in New York City that contain a lot of information about them such as their names, longitude, latitude, zip code and so on from <a href='https://hifld-geoplatform.opendata.arcgis.com/'>Homeland Infrastructure Foundation-Level Data (HIFLD)</a>, but from this dataset, I will need just the university name and latitude and longitude so, I will drop the other columns</li>
#   <li>Secondly, I will use the Foursquare Website to extract the trending venues around the 30 from the universities and academies with the help of the previously modified dataset of New York universities</li>
#     
# </ul>

# #### Importing the required libraries

# In[44]:


import pandas as pd # library for data analsysis
import requests # library to handle requests
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

# for clustering
from sklearn.cluster import KMeans
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# #### Importing and cleaning the universities data

# In[45]:


us_uni=pd.read_csv('Colleges_and_Universities.csv')


# In[46]:


NY_uni=pd.DataFrame(us_uni[us_uni["CITY"]=="NEW YORK"])
NY_uni.head()


# In[47]:


NY_uni.shape


# In[48]:


NY_uni.columns


# In[49]:


NY_uni.drop(columns=['X','OBJECTID', 'IPEDSID','Y','ADDRESS', 'CITY', 'STATE',
       'ZIP', 'ZIP4', 'TELEPHONE', 'TYPE', 'STATUS', 'POPULATION', 'COUNTY',
       'COUNTYFIPS', 'COUNTRY','NAICS_CODE',
       'NAICS_DESC', 'SOURCE', 'SOURCEDATE', 'VAL_METHOD', 'VAL_DATE',
       'WEBSITE', 'STFIPS', 'COFIPS', 'SECTOR', 'LEVEL_', 'HI_OFFER',
       'DEG_GRANT', 'LOCALE', 'CLOSE_DATE', 'MERGE_ID', 'ALIAS', 'SIZE_SET',
       'INST_SIZE', 'PT_ENROLL', 'FT_ENROLL', 'TOT_ENROLL', 'HOUSING',
       'DORM_CAP', 'TOT_EMP', 'SHELTER_ID'],inplace=True)
NY_uni.head()


# In[50]:


NY_uni.reset_index(drop=True,inplace=True)
NY_uni.head(57)


# In[75]:


CLIENT_ID = '####################################################' # Foursquare ID
CLIENT_SECRET = '##########################################' # Foursquare Secret
VERSION = '20180604'
LIMIT = 50
RADIUS= 2000
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# #### Creating a map of NYC's universities

# In[52]:


address = '102 North End Ave, New York, NY'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)


# In[53]:


# create map of New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, name in zip(NY_uni['LATITUDE'], NY_uni['LONGITUDE'], NY_uni['NAME']):
    label = '{}'.format(name)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_newyork)  
    
map_newyork


# # Getting the trending venues arounf each university

# ## CARSTEN INSTITUTE OF COSMETOLOGY

# In[54]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[14]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][1],NY_uni['LONGITUDE'][1], VERSION, RADIUS, LIMIT)

#url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}'.format(CLIENT_ID, CLIENT_SECRET,, VERSION)
results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    carsten_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    carsten_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[15]:


carsten_df


# In[17]:


df=pd.DataFrame()
df=df.append([NY_uni[1:2]]*4,ignore_index=True)
df=df.join(carsten_df [['name','categories','location.lat','location.lng']])
df


# ## AMERICAN MUSICAL AND DRAMATIC ACADEMY

# In[20]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][2],NY_uni['LONGITUDE'][2], VERSION, RADIUS, LIMIT)

#url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}'.format(CLIENT_ID, CLIENT_SECRET,, VERSION)
results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    musical_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    musical_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[21]:


musical_df


# In[152]:


df1=pd.DataFrame()
df1=df1.append([NY_uni[2:3]]*2,ignore_index=True)
df1=df1.join(musical_df [['name','categories','location.lat','location.lng']])
df1


# ## BERKELEY COLLEGE-NEW YORK

# In[23]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][3],NY_uni['LONGITUDE'][3], VERSION, RADIUS, LIMIT)

#url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}'.format(CLIENT_ID, CLIENT_SECRET,, VERSION)
results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    berkeley_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    berkeley_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[24]:


berkeley_df


# In[153]:


df2=pd.DataFrame()
df2=df2.append([NY_uni[3:4]]*3,ignore_index=True)
df2=df2.join(berkeley_df [['name','categories','location.lat','location.lng']])
df2


# ## CUNY SYSTEM OFFICE

# In[27]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][4],NY_uni['LONGITUDE'][4], VERSION, RADIUS, LIMIT)

#url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}'.format(CLIENT_ID, CLIENT_SECRET,, VERSION)
results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    cunyoff_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    cunyoff_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[28]:


cunyoff_df


# In[154]:


df3=pd.DataFrame()
df3=df3.append([NY_uni[4:5]]*3,ignore_index=True)
df3=df3.join(cunyoff_df [['name','categories','location.lat','location.lng']])
df3


# ## LIM COLLEGE

# In[36]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][7],NY_uni['LONGITUDE'][7], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    lim_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    lim_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[37]:


# display trending venues
lim_df


# In[38]:


df4=pd.DataFrame()
df4=df4.append([NY_uni[7:8]]*4,ignore_index=True)
df4=df4.join(lim_df[['name','categories','location.lat','location.lng']])
df4.head()


# ## THE NEW SCHOOL

# In[43]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][9],NY_uni['LONGITUDE'][9], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    news_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    news_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[44]:


news_df


# In[155]:


df5=pd.DataFrame()
df5=df5.append([NY_uni[9:10]]*1,ignore_index=True)
df5=df5.join(news_df[['name','categories','location.lat','location.lng']])
df5.head()


# ## TOURO COLLEGE

# In[47]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][10],NY_uni['LONGITUDE'][10], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    touro_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    touro_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[48]:


touro_df


# In[49]:


df6=pd.DataFrame()
df6=df6.append([NY_uni[10:11]]*4,ignore_index=True)
df6=df6.join(touro_df[['name','categories','location.lat','location.lng']])
df6


# ## MANHATTAN INSTITUTE

# In[55]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][13],NY_uni['LONGITUDE'][13], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    manhattan_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    manhattan_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[56]:


manhattan_df


# In[58]:


df7=pd.DataFrame()
df7=df7.append([NY_uni[13:14]]*3,ignore_index=True)
df7=df7.join(manhattan_df[['name','categories','location.lat','location.lng']])
df7


# ## INSTITUTE OF CULINARY EDUCATION

# In[63]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][14],NY_uni['LONGITUDE'][14], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    culinary_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    culinary_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[64]:


culinary_df


# In[156]:


df8=pd.DataFrame()
df8=df8.append([NY_uni[14:15]]*3,ignore_index=True)
df8=df8.join(culinary_df[['name','categories','location.lat','location.lng']])
df8


# ## FOCUS PERSONAL TRAINING INSTITUTE

# In[66]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][15],NY_uni['LONGITUDE'][15], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    focus_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    focus_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[67]:


focus_df


# In[68]:


df9=pd.DataFrame()
df9=df9.append([NY_uni[15:16]]*3,ignore_index=True)
df9=df9.join(focus_df[['name','categories','location.lat','location.lng']])
df9


# ## THE ART INSTITUTE OF NEW YORK CITY

# In[71]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][17],NY_uni['LONGITUDE'][17], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    art_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    art_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[72]:


art_df


# In[73]:


df10=pd.DataFrame()
df10=df10.append([NY_uni[17:18]]*4,ignore_index=True)
df10=df10.join(art_df[['name','categories','location.lat','location.lng']])
df10


# ## NEW AGE TRAINING

# In[74]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][18],NY_uni['LONGITUDE'][18], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    new_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    new_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[75]:


new_df


# In[76]:


df11=pd.DataFrame()
df11=df11.append([NY_uni[18:19]]*3,ignore_index=True)
df11=df11.join(new_df[['name','categories','location.lat','location.lng']])
df11


# ## THE JUILLIARD SCHOOL

# In[77]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][19],NY_uni['LONGITUDE'][19], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    juilliard_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    juilliard_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[78]:


juilliard_df


# In[157]:


df12=pd.DataFrame()
df12=df12.append([NY_uni[19:20]]*1,ignore_index=True)
df12=df12.join(juilliard_df[['name','categories','location.lat','location.lng']])
df12


# ## THE AILEY SCHOOL

# In[81]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][20],NY_uni['LONGITUDE'][20], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    ailey_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    ailey_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[82]:


ailey_df


# In[158]:


df13=pd.DataFrame()
df13=df13.append([NY_uni[20:21]]*3,ignore_index=True)
df13=df13.join(ailey_df[['name','categories','location.lat','location.lng']])
df13


# ## CUNY BERNARD M BARUCH COLLEGE

# In[85]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][21],NY_uni['LONGITUDE'][21], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    bernard_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    bernard_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[86]:


bernard_df


# In[87]:


df14=pd.DataFrame()
df14=df14.append([NY_uni[21:22]]*3,ignore_index=True)
df14=df14.join(bernard_df[['name','categories','location.lat','location.lng']])
df14


# ## MANDL SCHOOL-THE COLLEGE OF ALLIED HEALTH

# In[88]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][22],NY_uni['LONGITUDE'][22], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    mandl_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    mandl_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[89]:


mandl_df


# In[159]:


df15=pd.DataFrame()
df15=df15.append([NY_uni[22:23]]*3,ignore_index=True)
df15=df15.join(mandl_df[['name','categories','location.lat','location.lng']])
df15


# ## EMPIRE BEAUTY SCHOOL-MANHATTAN

# In[91]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][23],NY_uni['LONGITUDE'][23], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    empire_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    empire_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[92]:


empire_df


# In[160]:


df16=pd.DataFrame()
df16=df16.append([NY_uni[23:24]]*4,ignore_index=True)
df16=df16.join(empire_df[['name','categories','location.lat','location.lng']])
df16.head()


# ## SOTHEBY'S INSTITUTE OF ART-NY

# In[100]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][27],NY_uni['LONGITUDE'][27], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    sotheby_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    sotheby_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[101]:


sotheby_df


# In[161]:


df17=pd.DataFrame()
df17=df17.append([NY_uni[27:28]]*3,ignore_index=True)
df17=df17.join(sotheby_df[['name','categories','location.lat','location.lng']])
df17.head()


# ## CHRISTINE VALMY INTERNATIONAL SCHOOL

# In[109]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][30],NY_uni['LONGITUDE'][30], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    christine_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    christine_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[110]:


christine_df


# In[162]:


df18=pd.DataFrame()
df18=df18.append([NY_uni[30:31]]*3,ignore_index=True)
df18=df18.join(christine_df[['name','categories','location.lat','location.lng']])
df18.head()


# ## RELAY GRADUATE SCHOOL OF EDUCATION

# In[113]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][31],NY_uni['LONGITUDE'][31], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    relay_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    relay_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[114]:


relay_df


# In[163]:


df19=pd.DataFrame()
df19=df19.append([NY_uni[31:32]]*2,ignore_index=True)
df19=df19.join(relay_df[['name','categories','location.lat','location.lng']])
df19.head()


# ## DEVRY COLLEGE OF NEW YORK

# In[116]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][32],NY_uni['LONGITUDE'][32], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    devry_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    devry_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[117]:


devry_df


# In[164]:


df20=pd.DataFrame()
df20=df20.append([NY_uni[32:33]]*3,ignore_index=True)
df20=df20.join(devry_df[['name','categories','location.lat','location.lng']])
df20.head()


# ## CULINARY TECH CENTER

# In[120]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][33],NY_uni['LONGITUDE'][33], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    cultech_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    cultech_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[121]:


cultech_df


# In[165]:


df21=pd.DataFrame()
df21=df21.append([NY_uni[33:34]]*4,ignore_index=True)
df21=df21.join(cultech_df[['name','categories','location.lat','location.lng']])
df21.head()


# ## ATELIER ESTHETIQUE INSTITUTE OF ESTHETICS

# In[126]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][35],NY_uni['LONGITUDE'][35], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    atelier_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    atelier_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[127]:


atelier_df


# In[166]:


df22=pd.DataFrame()
df22=df22.append([NY_uni[35:36]]*3,ignore_index=True)
df22=df22.join(atelier_df[['name','categories','location.lat','location.lng']])
df22.head()


# ## CUNY HUNTER COLLEGE

# In[129]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][36],NY_uni['LONGITUDE'][36], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    cunyh_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    cunyh_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[130]:


cunyh_df


# In[167]:


df23=pd.DataFrame()
df23=df23.append([NY_uni[36:37]]*1,ignore_index=True)
df23=df23.join(cunyh_df[['name','categories','location.lat','location.lng']])
df23.head()


# ## TRI-STATE COLLEGE OF ACUPUNCTURE	

# In[132]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][37],NY_uni['LONGITUDE'][37], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    tri_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    tri_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[133]:


tri_df


# In[168]:


df24=pd.DataFrame()
df24=df24.append([NY_uni[37:38]]*1,ignore_index=True)
df24=df24.join(tri_df[['name','categories','location.lat','location.lng']])
df24.head()


# ## AMERICAN ACADEMY MCALLISTER INSTITUTE

# In[135]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][38],NY_uni['LONGITUDE'][38], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    mcallister_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    mcallister_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[136]:


mcallister_df


# In[169]:


df25=pd.DataFrame()
df25=df25.append([NY_uni[38:39]]*1,ignore_index=True)
df25=df25.join(mcallister_df[['name','categories','location.lat','location.lng']])
df25.head()


# ## GEMOLOGICAL INSTITUTE OF AMERICA-NEW YORK

# In[138]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][39],NY_uni['LONGITUDE'][39], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    gemo_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    gemo_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[139]:


gemo_df


# In[170]:


df26=pd.DataFrame()
df26=df26.append([NY_uni[39:40]]*3,ignore_index=True)
df26=df26.join(gemo_df[['name','categories','location.lat','location.lng']])
df26.head()


# ## NEW YORK UNIVERSITY

# In[171]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][40],NY_uni['LONGITUDE'][40], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    NYU_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    NYU_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[172]:


NYU_df


# In[173]:


df27=pd.DataFrame()
df27=df27.append([NY_uni[40:41]]*3,ignore_index=True)
df27=df27.join(NYU_df[['name','categories','location.lat','location.lng']])
df27.head()


# ## SWEDISH INSTITUTE A COLLEGE OF HEALTH SCIENCES

# In[144]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][41],NY_uni['LONGITUDE'][41], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    swedish_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    swedish_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[145]:


swedish_df


# In[147]:


df28=pd.DataFrame()
df28=df28.append([NY_uni[41:42]]*3,ignore_index=True)
df28=df28.join(swedish_df[['name','categories','location.lat','location.lng']])
df28.head()


# ## JOFFREY BALLET SCHOOL

# In[176]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][43],NY_uni['LONGITUDE'][43], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    joff_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    joff_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[177]:


joff_df


# In[179]:


df29=pd.DataFrame()
df29=df29.append([NY_uni[41:42]]*2,ignore_index=True)
df29=df29.join(joff_df[['name','categories','location.lat','location.lng']])
df29.head()


# ## COOPER UNION FOR THE ADVANCEMENT OF SCIENCE

# In[182]:


url = 'https://api.foursquare.com/v2/venues/trending?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET,NY_uni['LATITUDE'][45],NY_uni['LONGITUDE'][45], VERSION, RADIUS, LIMIT)

results = requests.get(url).json()
if len(results['response']['venues']) == 0:
    trending_venues_df = 'No trending venues are available at the moment!'
else:
    trending_venues = results['response']['venues']
    trending_venues_df = json_normalize(trending_venues)
    
    # filter columns
    columns_filtered = ['name', 'categories'] + ['location.distance', 'location.city', 'location.postalCode', 'location.state', 'location.country', 'location.lat', 'location.lng']
    cooper_df = trending_venues_df.loc[:, columns_filtered]
    
    # filter the category for each row
    cooper_df['categories'] = trending_venues_df.apply(get_category_type, axis=1)


# In[183]:


cooper_df


# In[184]:


df30=pd.DataFrame()
df30=df30.append([NY_uni[45:46]]*1,ignore_index=True)
df30=df30.join(joff_df[['name','categories','location.lat','location.lng']])
df30.head()


# ## Combining The Datasets Into One Dataset 

# In[185]:


dataset=pd.concat([df,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30])
dataset.reset_index(drop=True,inplace=True)
dataset


# ### Downloading the resulting dataframe

# In[190]:


from IPython.display import Javascript
js_download = """
var csv = '%s';

var filename = 'dataset.csv';
var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
if (navigator.msSaveBlob) { // IE 10+
    navigator.msSaveBlob(blob, filename);
} else {
    var link = document.createElement("a");
    if (link.download !== undefined) { // feature detection
        // Browsers that support HTML5 download attribute
        var url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}
""" % dataset.to_csv(index=False).replace('\n','\\n').replace("'","\'")

Javascript(js_download)


# In[191]:


import base64
import pandas as pd
from IPython.display import HTML

def create_download_link( dataset, title = "Download CSV file", filename = "data.csv"):
    csv = dataset.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


create_download_link(dataset)


# ## Explority Data Analysis

# In[55]:


df_data_0=pd.read_csv('new_data.csv')


# In[56]:


df_data_0.rename(columns={'NAME':'university','LATITUDE':'uni_lat','LONGITUDE':'uni_lng','name':'venue','location.lat':'venue_lat','location.lng':'venue_lng'},inplace=True)


# In[57]:


df_data_0.drop(columns='Unnamed: 0',inplace=True)


# The number of the venues category

# In[58]:


print('There are {} uniques categories.'.format(len(df_data_0['categories'].unique())))


# Making a data frame showing the names of the categories and how many venues are there and using it later for the visualization 

# In[59]:


grouped_by_categories=df_data_0.groupby('categories')['venue'].count()
grouped_by_categories=grouped_by_categories.to_frame()
grouped_by_categories


#  bar chart represent the categories and how many venues in each category

# In[60]:



ax = grouped_by_categories.plot.bar(y='venue',label=True)
ax
plt.title('Venues Catergories and their Frequency')
plt.ylabel('Frequency')


# And the pie chart to represent the categories of the venues by percentages 

# In[61]:


fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(8)
x=np.char.array(['Basketball Stadium','Grocery Store','Office','Park','Plaza','TV Station','Train Station'])
y= np.array([1,1,1,1,25,4,48])
patches, texts = plt.pie(y, startangle=90, radius=1.2)
porcent = 100.*y/y.sum()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)

plt.savefig('piechart.png', bbox_inches='tight')


# ## Analyzing The Dataset

# In[62]:


# one hot encoding
university_onehot = pd.get_dummies(df_data_0[['categories']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
university_onehot['university'] = df_data_0['university'] 

# move neighborhood column to the first column
fixed_columns = [university_onehot.columns[-1]] + list(university_onehot.columns[:-1])
university_onehot = university_onehot[fixed_columns]

university_onehot.head()


# #### grouping the rows by university and by taking the mean of the frequency of occurrence of each category

# In[63]:


uni_grouped = university_onehot.groupby('university').mean().reset_index()
uni_grouped


# #### Let's print each university along with the top 3 most common venues

# In[64]:


num_top_venues = 3

for university in uni_grouped['university']:
    print("----"+university+"----")
    temp = uni_grouped[uni_grouped['university'] == university].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# Making a Dataframe for that

# In[65]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[66]:


num_top_venues = 3

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['university']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
universities_venues_sorted = pd.DataFrame(columns=columns)
universities_venues_sorted['university'] = uni_grouped['university']

for ind in np.arange(uni_grouped.shape[0]):
    universities_venues_sorted.iloc[ind, 1:] = return_most_common_venues(uni_grouped.iloc[ind, :], num_top_venues)

universities_venues_sorted


# ## Cluster The Universities

# ### Finding the optimal number of clusters

# In[67]:


uni_grouped_clustering = uni_grouped.drop('university', 1)
Sum_of_squared_distances = []
K = range(1,8)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(uni_grouped_clustering)
    Sum_of_squared_distances.append(km.inertia_)


# In[68]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[69]:



# set number of clusters
kclusters = 4
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(uni_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# ### The clustered universities based on the common characteristics of the trending venues

# In[70]:


universities_venues_sorted.head()


# In[71]:


# add clustering labels
universities_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

data_merged = NY_uni

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
data_merged = data_merged.join(universities_venues_sorted.set_index('university'), on='NAME')

data_merged.head() # check the last columns!


# In[72]:


data_merged.fillna( method ='ffill', inplace = True) 
data_merged.drop(0,inplace=True)
data_merged.reset_index(drop=True,inplace=True)
data_merged['Cluster Labels'] =data_merged['Cluster Labels'].astype(int)
data_merged.head()


# In[73]:


universities_venues_sorted


# ### Map of the clustered universities based on the common characteristics of the trending venues

# In[74]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(data_merged['LATITUDE'], data_merged['LONGITUDE'], data_merged['NAME'], data_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:





# In[ ]:




