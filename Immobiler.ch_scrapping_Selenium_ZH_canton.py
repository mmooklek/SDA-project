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
# A) Scrap Immobilier.ch
# Import libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import re

# Set up the ChromeDriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode (without GUI)
driver_path = "C:/Users/Quentin/OneDrive/Studies/HSG/3rd semester/Smart Data Analytics/Project/chromedriver-win64/chromedriver.exe"  # Update this with the path to your chromedriver executable
service = ChromeService(executable_path=driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Load the webpage
base_url = "https://www.immobilier.ch/en/rent/apartment-house/zurich/page-"

# Function to extract data from a single listing
def scrape_listing(listing_url):
    driver.get(listing_url)
    
    try:
        # Set an explicit wait for elements to load
        wait = WebDriverWait(driver, 1)

        # Extract features as a list
        features = [feature.text.strip() for feature in driver.find_elements(By.CLASS_NAME, 'im__assets__table')]

        # Split features by '\n' and add to the data list
        features = '\n'.join(features)
        description = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'im__postContent__body'))).text.strip()

    except TimeoutException:
        features, description = [None] * 2

    return {
        "features": features,
        "description": description
    }

# Function to get the URLs of all listings on the current page
def get_listing_urls(page_number):
    url = f"{base_url}{page_number}?t=rent&c=1;2&p=c13660&nb=false&gr=1"
    driver.get(url)
    
    wait = WebDriverWait(driver, 1)  # Adjust the timeout as needed
    print(f"Waiting for page {page_number} to load...")
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "filter-item-container")))
    print("Element found!")
    
    # Locate the container element that holds the links
    listing_elements = driver.find_elements(By.CSS_SELECTOR, ".filter-item-container")
    
    listing_urls = []
    for element in listing_elements:
        try:
            listing_url = element.find_element(By.TAG_NAME, 'a').get_attribute('href')
            listing_urls.append(listing_url)
        except NoSuchElementException:
            print("No <a> tag found in the current container")
            continue
        
    return listing_urls

# Main scraping logic
try:
    # Create an empty DataFrame to store the data
    data = []
    
    # Loop through each page
    for page_number in range(1, 30):  # Change the range to iterate over all 40 pages
        # Get the URLs of all listings on the current page
        listing_urls = get_listing_urls(page_number)

        # Scrape data from each listing on the current page
        for listing_url in listing_urls:
            listing_data = scrape_listing(listing_url)

            # Split 'features' column by '\n' and create new columns for the first 10 features
            features_columns = listing_data["features"].split('\n')[:10]
            for i, value in enumerate(features_columns):
                listing_data[f"feature_{i + 1}"] = value

            # Remove the original 'features' column
            del listing_data["features"]

            data.append(listing_data)

    # Create a Pandas DataFrame
    df3 = pd.DataFrame(data)

    # Print the DataFrame
    print(df3)

finally:
    # Close the browser window
    driver.quit()
    
# saving the dataframe
df3.to_csv('Immobilier.ch_database_ZH_canton_071223.csv')

###############################################################################
# B) Align columns
# import dataset
df = pd.read_csv('Immobilier.ch_database_ZH_canton_071223.csv', index_col=0)

# Replace 'feature_3' and 123 (your integer substring) with the actual column name and integer substring you want to use
column_name = 'feature_3'
substring = 8

# Find the index of the 'feature_3' column
feature_3_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the integer value in the specified column does not start with the specified substring
    if not str(row[column_name]).startswith(str(substring)):
        # Shift values to the right starting from 'feature_3' column to the last column
        for i in range(len(row)-1, feature_3_index, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i-1]]
        
        # Set the value in the 'feature_3' column to None
        df.at[index, column_name] = None
    
###############################################################################
# Same for feature_5
column_name = 'feature_5'
substring = 'Costs'

# Find the index of the 'feature_5' column
feature_5_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the integer value in the specified column does not start with the specified substring
    if not str(row[column_name]).startswith(substring):
        # Shift values to the right starting from 'feature_3' column to the last column
        for i in range(len(row)-1, feature_5_index, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i-1]]
        
        # Set the value in the 'feature_5' column to None
        df.at[index, column_name] = None

###############################################################################    
# Same for feature_6
column_name = 'feature_6'
substring1 = 'rooms'
substring2 = 'm2'

# Find the index of the 'feature_6' column
feature_6_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column contains the specified substring
    if (substring1 in str(row[column_name])) or (substring2 in str(row[column_name])):
        # Shift values to the right starting from 'feature_3' column to the last column
        for i in range(len(row) - 1, feature_6_index, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i - 1]]

        # Set the value in the 'feature_6' column to None
        df.at[index, column_name] = None

###############################################################################
# shift feature_6 to the right
column_name = 'feature_6'
substring1 = 'Floor'
substring2 ='floor'

# Find the index of the 'feature_6' column
feature_6_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column contains the specified substring
    if (substring1 in str(row[column_name])) or (substring2 in str(row[column_name])):
        # Shift values to the right starting from 'feature_3' column to the last column
        for i in range(len(row) - 1, feature_6_index+1, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i - 2]]

        # Set the value in the 'feature_6' column to None
        df.at[index, column_name] = None
        df.at[index, df.columns[feature_3_index -1]] = None

df = df.drop(columns='feature_6')

###############################################################################     
# Replace 'feature_7' and 'feature_9' with the actual column names
column_name_7 = 'feature_7'
column_name_9 = 'feature_9'

for index, row in df.iterrows():
    # Check if values in 'feature_7' and 'feature_9' are duplicates within the same row
    if row[column_name_7] == row[column_name_9]:
        # Delete the value in 'feature_7' if they are duplicates
        df.at[index, column_name_7] = None

###############################################################################
# shift feature_7 two columns to the right if 'm2' is present
column_name = 'feature_7'
substring = 'm2'

# Find the index of the 'feature_7' column
feature_7_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column contains the specified substring
    if substring in str(row[column_name]):
        # Shift values to the right starting from 'feature_7' column to the last column
        for i in range(len(row) - 1, feature_7_index+1, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i - 2]]

        # Set the value in the 'feature_7' column to None
        df.at[index, column_name] = None
        df.at[index, df.columns[feature_7_index -1]] = None

###############################################################################
# Shift feature_8 to the right if 'm2' is present
column_name = 'feature_8'
substring = 'm2'

# Find the index of the 'feature_8' column
feature_8_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column contains the specified substring
    if substring in str(row[column_name]):
        # Shift values to the right starting from 'feature_8' column to the last column
        for i in range(len(row) - 1, feature_8_index, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i - 1]]

        # Set the value in the 'feature_8' column to None
        df.at[index, column_name] = None
        
###############################################################################
# Replace 'feature_8' and 'feature_9' with the actual column names
column_name_8 = 'feature_8'
column_name_10 = 'feature_10'

for index, row in df.iterrows():
    # Check if values in 'feature_8' and 'feature_10' are duplicates within the same row
    if row[column_name_8] == row[column_name_10]:
        # Delete the value in 'feature_8' if they are duplicates
        df.at[index, column_name_8] = None
        
###############################################################################        
# Delete content of the cell and shift to the left if 'Number' is present in feature_8
column_name = 'feature_8'
substring = 'Number'

# Find the index of the 'feature_8' column
feature_8_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column contains the specified substring
    if substring in str(row[column_name]):
        # Set the value in the 'feature_8' column to None
        df.at[index, column_name] = None
        
        # Shift values to the left starting from the column after 'feature_8'
        for i in range(feature_8_index + 1, len(row)):
            df.at[index, df.columns[i-1]] = df.at[index, df.columns[i]]

        # Set the value in the last column to None
        df.at[index, df.columns[-1]] = None

###############################################################################
# Shift feature_9 two columns to the right if 'm2' is NOT present
column_name = 'feature_9'
substring = 'm2'

# Find the index of the 'feature_9' column
feature_9_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column does NOT contain the specified substring
    if substring not in str(row[column_name]):
        # Shift values to the right starting from the last column to 'feature_9' column
        for i in range(len(row) - 1, feature_9_index - 1, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i - 2]]

        # Set the values in 'feature_9' and its preceding column to None
        df.at[index, column_name] = None
        df.at[index, df.columns[feature_9_index - 1]] = None
        
###############################################################################
# Shift feature_10 to the right if 'Built' is present
column_name = 'feature_10'
substring = 'Built'

# Find the index of the 'feature_10' column
feature_10_index = df.columns.get_loc(column_name)

for index, row in df.iterrows():
    # Check if the cell in the specified column contains the specified substring
    if substring not in str(row[column_name]):
        # Shift values to the right starting from 'feature_10' column to the last column
        for i in range(len(row) - 1, feature_10_index, -1):
            df.at[index, df.columns[i]] = df.at[index, df.columns[i - 1]]

        # Set the value in the 'feature_10' column to None
        df.at[index, column_name] = None

# save dataframe
df.to_csv('Immobilier.ch_database_ZH_canton_cleaned_071223.csv')

###############################################################################
# C) Clean data
# import dataset
df = pd.read_csv('Immobilier.ch_database_ZH_canton_cleaned_071223.csv', index_col=0)

df = df.rename(columns={'feature_1':'type','feature_2':'Address','feature_3':'Location',
                   'feature_4':'Rent','feature_5':'Costs','feature_7':'Rooms',
                   'feature_8':'Floor','feature_9':'Living space','feature_10':'Year'})

# Function to extract numeric values using regular expressions
def extract_numeric(value):
    # Use regex to extract the first group of digits (including dots for floats)
    matches = re.findall(r'([\d.]+)', str(value))
    # Convert the extracted string to a numeric value
    numeric_value = pd.to_numeric(matches[0], errors='coerce') if matches else None
    return numeric_value

def extract_rent(value):
    numeric_value = pd.to_numeric(''.join(filter(lambda x: x.isdigit() or x == '.', str(value))), errors='coerce')
    return numeric_value

# Apply the function to each column
df['Rent'] = df['Rent'].apply(extract_rent)
df['Costs'] = df['Costs'].apply(extract_numeric)
df['Rooms'] = df['Rooms'].apply(extract_numeric)
df['Living space'] = df['Living space'].apply(extract_numeric)
df['Year'] = df['Year'].apply(extract_numeric)

# Generate mapping dictionary for up to 20 floors
floor_mapping = {'Ground floor': 0, '1st floor': 1}
for i in range(2, 25):
    floor_mapping[f'Floor {i}'] = i

# Replace values in the 'floor' column
df['Floor'] = df['Floor'].replace(floor_mapping)

# Remove Google translation from the ...
df['description'] = df['description'].str.replace('Google translation from the original version \(DE\)', '', regex=True)

# Merge Address and location columns
df['Complete Address'] = df['Address']+', '+df['Location']
df = df.drop(columns=['Address','Location'])

# Save database
df.to_csv('Immobilier.ch_database_ZH_canton_final.csv')

# Save canton dataset
columns_to_test = ['Rent','Rooms','Living space','Complete Address']
df = df[df[columns_to_test].notna().all(axis=1)]
df = df.drop(columns=['Costs','Floor','Year'])
#df = pd.get_dummies(data=df, columns=['type'])
df.to_csv('Immobilier.ch_database_ZH_canton.csv')

###############################################################################
# C) Add spatial data
# import libraries
from geopy.geocoders import Nominatim
import time

df = df.reset_index(drop=True)

# transform address into coordinates
latitude = []
longitude = []
for i in range(len(df)):
    time.sleep(1)
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
gdf = gdf.to_crs("EPSG:4326")  # Make sure GeoDataFrame CRS is set to "EPSG:4326"
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

# City
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

# Building age
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

# Public transports ZH city
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

# Public transports stops zh
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

# Public parks in zh
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
    
# District
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