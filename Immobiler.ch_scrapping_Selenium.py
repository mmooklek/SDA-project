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
import re
import pandas as pd

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
    for page_number in range(1, 30):  # Change the range to iterate over all pages available
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
#df3.to_csv('Immobilier.ch_database_ZH_canton.csv')

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
#df.to_csv('Immobilier.ch_database_ZH_canton_cleaned.csv')

###############################################################################
# C) Clean data
# import dataset
#df = pd.read_csv('Immobilier.ch_database_ZH_canton_cleaned.csv', index_col=0)

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

# Save canton dataset
columns_to_test = ['Rent','Rooms','Living space','Complete Address']
df = df[df[columns_to_test].notna().all(axis=1)]
df = df.drop(columns=['Costs','Floor','Year'])
df.to_csv('Immobilier.ch_database_ZH_canton_final.csv')