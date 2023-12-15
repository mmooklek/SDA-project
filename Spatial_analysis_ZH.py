# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:43:06 2023
@author: Quentin
"""
###############################################################################
### HSG - Fall semester 2023
## Smart Data Analytics
# Project - Quentin Leoni & Jidapa Lengthaisong

###############################################################################
# C) Add spatial data
# import libraries
import pandas as pd
from geopy.geocoders import Nominatim
import time

df = pd.read_csv('Immobilier.ch_database_ZH_canton_final.csv')
df = df.reset_index(drop=True)

# transform address into coordinates
latitude = []
longitude = []
for i in range(len(df)):
    time.sleep(1) # KEEP IT TO 1 SECOND AT LEAST (limit of the free API)
    geolocator = Nominatim(user_agent="Smart Data Analytics - HSG")
    location = geolocator.geocode(df["Complete Address"][i])
    if location is None:
        latitude.append(None)
        longitude.append(None)
    else:
        latitude.append(float(location.latitude))
        longitude.append(float(location.longitude))
    
# Add longitude and latitude to the dataframe
df['latitude'] = latitude
df['longitude'] = longitude

# Save temporary subset
#df.to_csv('Immobilier.ch_database_ZH_canton_CRS.csv')
df = pd.read_csv('Immobilier.ch_database_ZH_canton_CRS.csv')

###############################################################################
# Add spatial data
# import libraries
import geopandas
import matplotlib.pyplot as plt

# Construct a geodataframe for the canton
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.longitude, df.latitude, crs="EPSG:4326"))

# Load canton shapefile
city = geopandas.read_file("ZH canton/Gemeindegrenzen/data/UP_GEMEINDEN_F.shp")
public_transport_stops = geopandas.read_file("ZH canton/Haltestellen des öffentlichen Verkehrs/data/ZVV_HALTESTELLEN_P.shp")
public_transport_lines_G = geopandas.read_file("ZH canton/Linien des öffentlichen Verkehrs/data/ZVV_LINIEN_GEN_L.shp")
public_transport_lines_S = geopandas.read_file("ZH canton/Linien des öffentlichen Verkehrs/data/ZVV_S_BAHN_LINIEN_L.shp")
building_age = geopandas.read_file("ZH canton/Gebäudealter/data/GEBAEUDEALTER_P.shp")
# Add Zurich maps
district = geopandas.read_file("ZH/Statistiche Quartiere/data/stzh.adm_statistische_quartiere_map.shp")
public_transport_stops_zh = geopandas.read_file("ZH/Haltestellen VBZ/data/ZVV_HALTESTELLEN_P.shp")
public_transport_lines_G_zh = geopandas.read_file("ZH/Linien des öffentlichen Verkehrs/data/ZVV_LINIEN_GEN_L.shp")
public_transport_lines_S_zh = geopandas.read_file("ZH/Linien des öffentlichen Verkehrs/data/ZVV_S_BAHN_LINIEN_L.shp")
parks = geopandas.read_file("ZH/Grünflächen/data/gsz.gruenflaechen.shp")
gardens = geopandas.read_file("ZH/Familiengärten/data/stzh.poi_familiengarten_view.shp")
bikes = geopandas.read_file("ZH/Standorte ZüriVelo/data/taz.view_zuerivelo_publibike.shp")

# Ensure CRS matches
#gdf = gdf.to_crs("EPSG:4326")  # Make sure GeoDataFrame CRS is set to "EPSG:4326"
city = city.to_crs("EPSG:4326")
public_transport_stops = public_transport_stops.to_crs("EPSG:4326")
public_transport_lines_G = public_transport_lines_G.to_crs("EPSG:4326")
public_transport_lines_S = public_transport_lines_S.to_crs("EPSG:4326")
public_transport_lines_G_zh = public_transport_lines_G_zh.to_crs("EPSG:4326")
public_transport_lines_S_zh = public_transport_lines_S_zh.to_crs("EPSG:4326")
building_age = building_age.to_crs("EPSG:4326")
district = district.to_crs("EPSG:4326")
public_transport_stops_zh = public_transport_stops_zh.to_crs("EPSG:4326")
parks = parks.to_crs("EPSG:4326")
gardens = gardens.to_crs("EPSG:4326")
bikes = bikes.to_crs("EPSG:4326")

# Check overlapping geometries
data = gdf[gdf.geometry.intersects(city.unary_union)]
data_zh = gdf[gdf.geometry.intersects(district.unary_union)]

###############################################################################
# Function to calculate distance to the nearest point in 'gardens'
def calculate_nearest_distance(row, gardens):
    distances = gardens.geometry.distance(row['geometry'])
    min_distance = distances.min()
    return min_distance

# Apply the function to each row in the 'data' GeoDataFrame
data['Public_transport'] = data.apply(lambda row: calculate_nearest_distance(row, public_transport_stops), axis=1)

# Function to return the year of the building (nearest point in building_age)
def calculate_nearest_GBAUJ(row, building_age):
    distances = building_age.geometry.distance(row['geometry'])
    min_distance_idx = distances.idxmin()
    nearest_GBAUJ = building_age.at[min_distance_idx, 'GBAUJ']
    return nearest_GBAUJ

# Add a new column 'nearest_GBAUJ' to store the GBAUJ values of the nearest geometries
data['Year'] = data.apply(calculate_nearest_GBAUJ, building_age=building_age, axis=1)

# Create a new column 'Neighborhood' and set the default value to None
data['City'] = None

# Iterate through each polygon and assign the neighborhood name
for index, row in city.iterrows():
    city_name = row['GEMEINDENA']
    data.loc[data['geometry'].within(row['geometry']), 'City'] = city_name

#data.to_csv('Immobilier.ch_database_ZH_canton_completed.csv')

data = data.drop(columns=['Complete Address', 'latitude','longitude'])
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#data.to_csv('Immobilier.ch_database_ZH_7_cov.csv')
data = pd.read_csv('Immobilier.ch_database_ZH_7_cov.csv')   

###############################################################################
# Create dataset for the city

# Apply the function to each row in the 'data' GeoDataFrame
data_zh['Gardens'] = data_zh.apply(lambda row: calculate_nearest_distance(row, gardens), axis=1)
data_zh['PubliBike'] = data_zh.apply(lambda row: calculate_nearest_distance(row, bikes), axis=1)
data_zh['Park'] = data_zh.apply(lambda row: calculate_nearest_distance(row, parks), axis=1)
data_zh['Public_transport'] = data_zh.apply(lambda row: calculate_nearest_distance(row, public_transport_stops_zh), axis=1)
data_zh['Year'] = data_zh.apply(calculate_nearest_GBAUJ, building_age=building_age, axis=1)

# Create a new column 'Neighborhood' and set the default value to None
data_zh['District'] = None

# Iterate through each polygon and assign the neighborhood name
for index, row in district.iterrows():
    city_name = row['qname']
    data_zh.loc[data_zh['geometry'].within(row['geometry']), 'District'] = city_name
    
data_zh = data_zh.drop(columns=['Complete Address', 'latitude','longitude'])
data_zh = data_zh.loc[:, ~data_zh.columns.str.contains('^Unnamed')]
#data_zh.to_csv('Immobilier.ch_database_ZH_9_cov.csv')
data_zh = pd.read_csv('Immobilier.ch_database_ZH_9_cov.csv')

###############################################################################
# Create a plot function
def plot(file,data,color_file,edgecolor_file,color_data):
    fig, ax = plt.subplots(figsize=(50, 50), dpi=100)
    district.plot(ax=ax, color='lightgrey', edgecolor='black')
    file.plot(ax=ax, color=color_file, edgecolor=edgecolor_file, markersize=100)
    data.plot(ax=ax, color=color_data, markersize=100)
    plt.show()

plot(gardens,data_zh,'green','green','red')
plot(bikes,data_zh,'blue','blue','red')

# create new column based on the price per sqm
data['CHF/m2/year'] = data['Rent']*12/data['Living space']
Rent_level = data['CHF/m2/year'].groupby(data['City']).mean()
city['City'] = city['GEMEINDENA']
city = city.drop(columns='GEMEINDENA')
city = geopandas.GeoDataFrame(pd.merge(Rent_level, city, on='City', how='right'))

# Cities in canton
fig, ax = plt.subplots(figsize=(50,50), dpi=100)
city.plot(ax=ax, color='lightgrey', edgecolor='black')
city.plot(ax=ax,
          column='CHF/m2/year',
          figsize=(100,100),
          markersize=100,
          legend=True,
          cmap='coolwarm',
          legend_kwds={"label": "ZH canton municipalities", "orientation": "vertical"})
#data.plot(ax=ax, color='red', markersize=50)
plt.show()

# Building age canton
fig, ax = plt.subplots(figsize=(50, 50), dpi=100)
city.plot(ax=ax, color='white',edgecolor='black')
building_age.plot(ax=ax,
                  column='GBAUJ',
                  figsize=(50,50),
                  markersize=1,
                  legend=True,
                  scheme='quantiles')
plt.show()

# Public transports canton
fig, ax = plt.subplots(figsize=(50, 50), dpi=100)
public_transport_lines_S.plot(ax=ax,
                              column='LINIESBAHN',
                              figsize=(100,100),
                              markersize=150,
                              legend=True)
city.plot(ax=ax, color='lightgrey',edgecolor='lightgrey')
city[city['City']=='Zürich'].plot(ax=ax, color='white')
#data.plot(ax=ax, color='red', markersize=50)
plt.show()

# Public transports in ZH city
fig, ax = plt.subplots(figsize=(50, 50), dpi=100)
public_transport_lines_G_zh[public_transport_lines_G_zh['RICHTUNG']==1].plot(ax=ax,
                                 column='LINIENNUMM',
                                 figsize=(100,100),
                                 markersize=50,
                                 legend=True)
district.plot(ax=ax, color='lightgrey',edgecolor='lightgrey')
#data_zh.plot(ax=ax, color='red', markersize=50)
plt.show()

# Public transports stops canton
public_transport_stops = public_transport_stops[public_transport_stops.geometry.intersects(city.unary_union)]

fig, ax = plt.subplots(figsize=(50,50), dpi=100)
city.plot(ax=ax, color='lightgrey', edgecolor='white')
public_transport_stops.plot(ax=ax,
                            column='SYMB_TEXT',
                            figsize=(100,100),
                            markersize=50,
                            legend=True)
#data.plot(ax=ax, color='red', markersize=50)
plt.show()

# Public transports stops in ZH city
public_transport_stops_zh = public_transport_stops_zh[public_transport_stops_zh.geometry.intersects(district.unary_union)]

fig, ax = plt.subplots(figsize=(50,50), dpi=100)
district.plot(ax=ax, color='lightgrey', edgecolor='white')
public_transport_stops_zh.plot(ax=ax,
                               column='SYMB_TEXT',
                               figsize=(100,100),
                               markersize=100,
                               legend=True)
#data.plot(ax=ax, color='red', markersize=50)
plt.show()

# Public parks in ZH city
fig, ax = plt.subplots(figsize=(50,50), dpi=100)
district.plot(ax=ax, color='white', edgecolor='black')
parks.plot(ax=ax,color='green')
data_zh.plot(ax=ax, color='red', markersize=100)
plt.show()

# create new column based on the price per sqm
data_zh['CHF/m2/year'] = data_zh['Rent']*12/data_zh['Living space']
Rent_level_zh = data_zh['CHF/m2/year'].groupby(data_zh['District']).mean()
district['District'] = district['qname']
district = district.drop(columns='qname')
district = geopandas.GeoDataFrame(pd.merge(Rent_level_zh, district, on='District', how='right'))
    
# District in ZH city
fig, ax = plt.subplots(figsize=(50,50), dpi=100)
district.plot(ax=ax, color='lightgrey', edgecolor='black')
district.plot(ax=ax,
              column='CHF/m2/year',
              figsize=(100,100),
              markersize=100,
              legend=True,
              cmap='coolwarm',
              legend_kwds={'shrink': 0.5})
#data_zh.plot(ax=ax, color='red', markersize=50)
plt.show()