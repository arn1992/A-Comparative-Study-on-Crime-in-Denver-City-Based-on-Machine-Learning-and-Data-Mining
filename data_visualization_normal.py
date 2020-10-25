import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import timedelta
from main import compare_columns
import folium
from folium.plugins import HeatMap
from folium import plugins
import seaborn as sns

df = pd.read_csv('./data/crime.csv')
'''
oldest_date = df['REPORTED_DATE'].min()
newest_date = df['REPORTED_DATE'].max()
print(oldest_date)
print(newest_date)
print(min(df['REPORTED_DATE']),max(df['REPORTED_DATE']))
'''
print(df.OFFENSE_CATEGORY_ID=='traffic-accident')
print(df.dtypes)
print("IS_CRIME: ", df['IS_CRIME'].nunique())
print("IS_TRAFFIC: ", df['IS_TRAFFIC'].nunique())
print("District ID: ", df['DISTRICT_ID'].nunique())
print("PRECINCT_ID: ", df['PRECINCT_ID'].nunique())
print("OFFENSE_CODE_EXTENSION", df['OFFENSE_CODE_EXTENSION'].nunique())


print("nunique: ",df['OFFENSE_CATEGORY_ID'].nunique())
print("nunique: ",df['OFFENSE_TYPE_ID'].nunique())
df['REPORTED_DATE'] = pd.to_datetime(df['REPORTED_DATE'])
least_recent_date = df['REPORTED_DATE'].min()
recent_date = df['REPORTED_DATE'].max()
print(least_recent_date,recent_date)
#total crimes by category

a= df.groupby('OFFENSE_CATEGORY_ID')['OFFENSE_CATEGORY_ID'].count()

print(a)
b=df.groupby('OFFENSE_TYPE_ID')['OFFENSE_TYPE_ID'].count()
print(b)
# Get a count of the top 20 crimes, based on the "OFFENSE_TYPE_ID"
crime_cnts = df[['OFFENSE_CATEGORY_ID','OFFENSE_ID']].groupby(['OFFENSE_CATEGORY_ID'],as_index=False).count().nlargest(15,['OFFENSE_ID'])

# Plot the most common crimes

df.OFFENSE_CATEGORY_ID.value_counts().plot(kind='barh', fontsize=8,legend=False)
plt.show()

# Which hours of the day are getting the most traffic crimes
# Group all traffic crimes into a data set
traffic = df[df['OFFENSE_TYPE_ID'].str[:4] == 'traf']
traffic['HOUR'] = pd.DatetimeIndex(traffic['REPORTED_DATE']).hour
traf_hour = traffic[['OFFENSE_TYPE_ID','HOUR']].groupby(['HOUR'],as_index=False).count()
ax = traf_hour.plot(kind='bar', x='HOUR', title ="Which Hour of the Day is Getting the Most Traffic Crimes", figsize=(15, 8), fontsize=12,legend=False,)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Crime/Hour", fontsize=12)
plt.show()
'''
districts = sorted(df['DISTRICT_ID'].unique())

#fig1, ax = plt.subplots(figsize=(10,5))

df.groupby('OFFENSE_CATEGORY_ID').DISTRICT_ID.value_counts().sort_index().unstack().plot(kind='barh', fontsize=8,legend=False)

#ax.legend(title='District', fontsize=20)
#figs_save['Crime by category and district'] = (fig1, './data/crime_categ_dist.jpg')
plt.savefig('./figure/crime_categ_dist.png')
plt.show()

df['FIRST_OCCURRENCE_DATE'] = pd.to_datetime(df['FIRST_OCCURRENCE_DATE'], errors='coerce')
oldest_date = df['FIRST_OCCURRENCE_DATE'].min()
newest_date = df['FIRST_OCCURRENCE_DATE'].max()
print(newest_date)

# mask2019 = df_crime['FIRST_OCCURRENCE_DATE'] > datetime(year=2018, month=12, day=31, hour=23, minute=59, second=59)
mask2019 = df.FIRST_OCCURRENCE_DATE.dt.year == 2019
df[~mask2019].FIRST_OCCURRENCE_DATE.max()
print(df[~mask2019].FIRST_OCCURRENCE_DATE.max())

fig, ax = plt.subplots(figsize=(11,6))
title = 'Crimes per month'

df_year_month = df[~mask2019].groupby([df.FIRST_OCCURRENCE_DATE.dt.year,
                                             df.FIRST_OCCURRENCE_DATE.dt.month]
                                           ).INCIDENT_ID.count().unstack(0)
df_year_month.rename_axis('Year', axis='columns',inplace=True)
df_year_month.index.rename('Month', inplace=True)

df_year_month.plot(xticks=range(1,13), ax=ax, linewidth=3)

ax.set_title(title)
ax.set_xlabel('Month')
ax.set_ylabel('Crime count')
ax.legend(loc='upper right')
ax.set_xticklabels(('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'))
plt.savefig('./figure/month_by_crime.png')
plt.show()

fig, ax = plt.subplots(figsize=(15,8))
title = 'Crime Categories by Hour'

time_cat_count = df[~mask2019
                         ].groupby(['OFFENSE_CATEGORY_ID',
                                    df[~mask2019].FIRST_OCCURRENCE_DATE.dt.hour]
                                  ).INCIDENT_ID.count().unstack(0)
time_cat_count.index.rename('Hour', inplace=True)

time_cat_count.plot(ax=ax,
                    cmap='tab20',
                    title=title,
                    linewidth=3,
                    xticks=range(24)
                   )


ax.legend(loc='upper right')
ax.set_xlabel('Time of day: Hour')
ax.set_ylabel('Crime count')
# ax.set_title(title)
#figs_save[title] = (fig, 'crime_cate_hour.png')
plt.savefig('./figure/hour_by_crime.png')
plt.show()

'''
# Captures 504,098 out of 508,459 rows of data (99%). The rest was outliers and/or misclassified.
'''
df = df[(df['GEO_LON'] < -50) & (df['GEO_LAT'] > 38)]

plt.figure(figsize=(12,10))
ax = sns.scatterplot(x='GEO_LON',y='GEO_LAT', data=df)
plt.show()

## district separation ##
plt.figure(figsize=(10,10))
sns.scatterplot(x='GEO_LON',
                y='GEO_LAT',
                alpha=0.5,
                hue='DISTRICT_ID',
                palette=plt.get_cmap('jet'),
                legend='full',
                data=df)
plt.show()

offense_cats = df['OFFENSE_CATEGORY_ID'].value_counts()[:14].index

plt.figure(figsize=(12,10))
sns.scatterplot(x='GEO_LON',
                y='GEO_LAT',
                hue='OFFENSE_CATEGORY_ID',
                data=df[df['OFFENSE_CATEGORY_ID'].isin(offense_cats)])
plt.show()


# Add mapping libraries and traffic summaries on a geographic map


map_den = folium.Map(location= [39.76,-105.02], zoom_start = 16)

# Get data from 15:00 to 18:00
den15_18 = traffic[(traffic['HOUR'] >= 15) & (traffic['HOUR'] <= 18)]

# Create a list with lat and long values and add the list to a heat map, then show map
heat_data = [[row['GEO_LAT'],row['GEO_LON']] for index, row in den15_18.iterrows()]
HeatMap(heat_data).add_to(map_den)

map_den.save('heatmap.html')
'''

'''
year2018 = (df.REPORTED_DATE >= '2017-01-01') & (df.REPORTED_DATE < '2018-01-01')
data2018 = df[(year2018) & (df.DISTRICT_ID == 1) & (df.OFFENSE_TYPE_ID =='theft-shoplift')]
idx = data2018['GEO_LAT'].isna() | data2018['GEO_LON'].isna()
data2018 = data2018[~idx]
m = folium.Map(location=[39.76,-105.02], tiles='Stamen Toner',zoom_start=13, control_scale=True)
from folium.plugins import MarkerCluster
mc = MarkerCluster()
for each in data2018.iterrows():
    mc.add_child(folium.Marker(location = [each[1]['GEO_LAT'],each[1]['GEO_LON']]))
m.add_child(mc)
m.save('ui1.html')
'''
df.FIRST_OCCURRENCE_DATE = pd.to_datetime(df.FIRST_OCCURRENCE_DATE)
df["YEAR"] = df.FIRST_OCCURRENCE_DATE.dt.year
df = df[df.IS_CRIME==1]
m=df.OFFENSE_CATEGORY_ID=='traffic-accident'
traffic_df = df.dropna(subset=['GEO_LAT', 'GEO_LON'])
traffic_df = traffic_df[(m)& (df.DISTRICT_ID == 3) & (traffic_df["YEAR"])]
d_map = folium.Map(location=[39.72378, -104.899157],
                       zoom_start=12,
                       tiles="OpenStreetMap")
for i in range(len(traffic_df)):
    lat = traffic_df.iloc[i]['GEO_LAT']
    long = traffic_df.iloc[i]['GEO_LON']
    popup_text = """Neighborhood: {}<br>
                    Date Occurred: {}<br>""".format(traffic_df.iloc[i]['NEIGHBORHOOD_ID'],
                                               traffic_df.iloc[i]['FIRST_OCCURRENCE_DATE'])
    folium.CircleMarker(location=[lat, long], popup=popup_text, radius=2, color='blue', fill=True).add_to(d_map)
d_map.save('traffict_accident_district_3.html')

