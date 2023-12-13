"""
Smart Data Analytics

Fall Semester 2023.

University of St. Gallen.

Jidapa Lengthaisong (22-601-355) and Quentin Leoni (19-406-560)

"""

# import modules here
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plot
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import seaborn as sns



# load own functions
import Final_Project_Group_1_Functions as functions


# ensure that PATH coincides with your working directory.
PATH = os.getcwd()

# define the name for the output file
OUTPUT_NAME = 'Final Project_Group 1'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = functions.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME7 = 'Immobilier.ch_database_ZH_7_cov.csv' # Data for Canton of Zurich (ZH)
DATANAME9 = 'Immobilier.ch_database_ZH_9_cov.csv' # Data for City of Zurich (CZH)

# load in data using pandas
data_zh = pd.read_csv(PATH + '/' + DATANAME7)
data_czh = pd.read_csv(PATH + '/' + DATANAME9)

# remove unwanted column
data_zh.drop(columns = ['Unnamed: 0'], inplace=True) 
data_czh.drop(columns = ['Unnamed: 0'], inplace=True)

# Create new variable 
data_zh['price_per_sqm'] = (data_zh['Rent']*12)/data_zh['Living space']
data_czh['price_per_sqm'] = (data_czh['Rent']*12)/data_zh['Living space']

# Remove nan from Year Column
data_zh = data_zh.dropna(subset=['Year'])

# Save cleaned data
data_zh.to_csv('Immobilier.ch_database_ZH_9_cov_cleaned.csv', index=False)
data_czh.to_csv('Immobilier.ch_database_ZH_9_cov_cleaned.csv', index=False) 

###############################################################################

# Exploratory Data Analysis (EDA) 

###############################################################################
# select continuous variables
continuous_vars_zh = ['Rent', 'Rooms', 'Living space','Public_transport','Year','price_per_sqm']
continuous_vars_czh = ['Rent', 'Rooms', 'Living space', 'Gardens', 'PubliBike','Park','Public_transport','Year','price_per_sqm']

# (1.) Inspect the both datasets and report descriptive statistics
data_zh.info()
data_czh.info()
functions.my_summary_stats(data_zh[continuous_vars_zh])
functions.my_summary_stats(data_czh[continuous_vars_czh])



# (2.) Histrograms
# Canton of Zurich
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars_zh):
    plt.subplot(3, 3, i+1)  # Adjust the grid size according to the number of variables
    plt.hist(data_zh[var].dropna(), bins=20, edgecolor='black')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# City of Zurich
plt.figure(figsize=(15, 10))
for i, var in enumerate(continuous_vars_czh):
    plt.subplot(3, 3, i+1)  # Adjust the grid size according to the number of variables
    plt.hist(data_czh[var].dropna(), bins=20, edgecolor='black')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



# (3.) QQ-plot
# Canton of Zurich
fig_zh = sm.qqplot(data_zh['Rent'], line ='45', fit=True)

# Customize the plot with matplotlib
plot.title("QQ Plot for Canton of Zurich")
plot.xlabel("Theoretical Quantiles")
plot.ylabel("Sample Quantiles")
plot.gca().lines[1].set_color("black")
plot.gca().get_lines()[0].set_color("steelblue")
plot.gca().get_lines()[0].set_markerfacecolor("steelblue")
plot.gca().get_lines()[0].set_markeredgecolor("steelblue")
plot.show()


# City of Zurich
fig_czh = sm.qqplot(data_czh['Rent'], line ='45', fit=True)

# Customize the plot with matplotlib
plot.title("QQ Plot for City of Zurich")
plot.xlabel("Theoretical Quantiles")
plot.ylabel("Sample Quantiles")
plot.gca().lines[1].set_color("black")
plot.gca().get_lines()[0].set_color("steelblue")
plot.gca().get_lines()[0].set_markerfacecolor("steelblue")
plot.gca().get_lines()[0].set_markeredgecolor("steelblue")
plot.show()


# (4.) Correlation Heatmap
functions.correlation_heatmap(data_zh[list(data_zh.columns[2:7])])
czh_columns = list(data_czh.columns[2:5]) + list(data_czh.columns[6:11])
functions.correlation_heatmap(data_czh[czh_columns])


# (5.) Boxplot of all continuous variables 
# Canton of Zurich
plt.figure(figsize=(15, 10))

# Loop through the variables and create a boxplot for each
for i, var in enumerate(continuous_vars_zh):
    # Calculate row and column index for the subplot
    row_idx = i // 3  # Integer division to get row index (0 or 1)
    col_idx = i % 3   # Modulus to get column index (0, 1, or 2)

    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=data_zh[var])
    plt.title(f'Boxplot of {var}')
    plt.xlabel(var)

plt.tight_layout()
plt.show()

# City of Zurich
plt.figure(figsize=(15, 15))

# Loop through the variables and create a boxplot for each
for i, var in enumerate(continuous_vars_czh):
    # Calculate row and column index for the subplot
    row_idx = i // 3  # Integer division to get row index (0 or 1)
    col_idx = i % 3   # Modulus to get column index (0, 1, or 2)

    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=data_czh[var])
    plt.title(f'Boxplot of {var}')
    plt.xlabel(var)

plt.tight_layout()
plt.show()


# (6.) Scatterplot for every continuous variables against prices
# Canton of Zurich
for var in continuous_vars_zh:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_zh[var], y=data_zh['Rent'])
    plt.title(f'Scatterplot of Rent vs. {var}')
    plt.xlabel(var)
    plt.ylabel('Rent')
    plt.show()

# City of Zurich
for var in continuous_vars_czh:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_czh[var], y=data_czh['Rent'])
    plt.title(f'Scatterplot of Rent vs. {var}')
    plt.xlabel(var)
    plt.ylabel('Rent')
    plt.show()



# (7.) Pairwise relationship
# Canton of Zurich
sns.pairplot(data_zh[continuous_vars_zh], plot_kws={'color': 'steelblue'})
plt.show()

# City of Zurich
sns.pairplot(data_czh[continuous_vars_czh], plot_kws={'color': 'steelblue'})
plt.show()




# (8.) Rental price by neighborhood group
# Canton of Zurich
encoded_data_zh = pd.get_dummies(data_zh, columns=['City'])
neighborhood_cols_zh = encoded_data_zh.columns[8:]

# Preparing data for the bar chart
df_melted_zh = encoded_data_zh.melt(id_vars=['Rent'], value_vars=neighborhood_cols_zh, var_name='Neighborhood', value_name='Presence')
df_melted_zh = df_melted_zh[df_melted_zh['Presence'] == 1]
# Group by neighborhood and calculate the average rent
average_rent_zh = df_melted_zh.groupby('Neighborhood')['Rent'].mean()

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_ZH = 6

# Group by neighborhood, calculate the average rent and count of listings
avg_rent_count_zh = df_melted_zh.groupby('Neighborhood').agg({'Rent': ['mean', 'count']})

# Filter out neighborhoods with less than the threshold number of listings
filtered_avg_rent_zh = avg_rent_count_zh[avg_rent_count_zh[('Rent', 'count')] >= MIN_LISTINGS_THRESHOLD_ZH]

# Sorting the average rent values in descending order
average_rent_sorted_zh = filtered_avg_rent_zh[('Rent', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_zh.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price by Neighborhood in the Canton of Zurich')
plt.xlabel('Neighborhood')
plt.ylabel('Average Rental Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# City of Zurich
encoded_data_czh = pd.get_dummies(data_czh, columns=['District'])
neighborhood_cols_czh = encoded_data_czh.columns[12:]

# Preparing data for the bar chart
df_melted_czh = encoded_data_czh.melt(id_vars=['Rent'], value_vars=neighborhood_cols_czh, var_name='Neighborhood', value_name='Presence')
df_melted_czh = df_melted_czh[df_melted_czh['Presence'] == 1]
# Group by neighborhood and calculate the average rent
average_rent_czh = df_melted_czh.groupby('Neighborhood')['Rent'].mean()

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_czh = 6

# Group by neighborhood, calculate the average rent and count of listings
avg_rent_count_czh = df_melted_czh.groupby('Neighborhood').agg({'Rent': ['mean', 'count']})

# Filter out neighborhoods with less than the threshold number of listings
filtered_avg_rent_czh = avg_rent_count_czh[avg_rent_count_czh[('Rent', 'count')] >= MIN_LISTINGS_THRESHOLD_czh]

# Sorting the average rent values in descending order
average_rent_sorted_czh = filtered_avg_rent_czh[('Rent', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_czh.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price by Neighborhood in the Canton of Zurich')
plt.xlabel('Neighborhood')
plt.ylabel('Average Rental Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# (9.) Rental price per squared meter by neighborhood group
# Canton of Zurich
df_melted = encoded_data_zh.melt(id_vars=['price_per_sqm'], value_vars=neighborhood_cols_zh, var_name='Neighborhood', value_name='Presence')
df_melted = df_melted[df_melted['Presence'] == 1]
# Group by neighborhood and calculate the average rent
average_rent = df_melted.groupby('Neighborhood')['price_per_sqm'].mean()

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_ZH = 6

# Group by neighborhood, calculate the average rent and count of listings
avg_rent_count_zh = df_melted.groupby('Neighborhood').agg({'price_per_sqm': ['mean', 'count']})

# Filter out neighborhoods with less than the threshold number of listings
filtered_avg_rent_zh = avg_rent_count_zh[avg_rent_count_zh[('price_per_sqm', 'count')] >= MIN_LISTINGS_THRESHOLD_ZH]

# Sorting the average rent values in descending order
average_rent_sorted_zh = filtered_avg_rent_zh[('price_per_sqm', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_zh.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price per sqm by Neighborhood in the Canton of Zurich')
plt.xlabel('Neighborhood')
plt.ylabel('Average Rental Price per sqm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# City of Zurich
df_melted2 = encoded_data_czh.melt(id_vars=['price_per_sqm'], value_vars=neighborhood_cols_czh, var_name='Neighborhood', value_name='Presence')
df_melted2 = df_melted2[df_melted2['Presence'] == 1]
# Group by neighborhood and calculate the average rent
average_rent2 = df_melted2.groupby('Neighborhood')['price_per_sqm'].mean()

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_czh = 6

# Group by neighborhood, calculate the average rent and count of listings
avg_rent_count_czh = df_melted2.groupby('Neighborhood').agg({'price_per_sqm': ['mean', 'count']})

# Filter out neighborhoods with less than the threshold number of listings
filtered_avg_rent_czh = avg_rent_count_czh[avg_rent_count_czh[('price_per_sqm', 'count')] >= MIN_LISTINGS_THRESHOLD_czh]

# Sorting the average rent values in descending order
average_rent_sorted_czh = filtered_avg_rent_czh[('price_per_sqm', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_czh.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price per sqm by Neighborhood in the City of Zurich')
plt.xlabel('Neighborhood')
plt.ylabel('Average Rental Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# (10.) Rental price by type
# Canton of Zurich
encoded_data_zht = pd.get_dummies(data_zh, columns=['type'])
type_cols_zht = encoded_data_zht.columns[8:]

# Preparing data for the bar chart
df_melted_zht = encoded_data_zht.melt(id_vars=['Rent'], value_vars=type_cols_zht, var_name='Type', value_name='Presence')
df_melted_zht = df_melted_zht[df_melted_zht['Presence'] == 1]

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_ZHT = 3

# Group by type, calculate the average rent and count of listings
avg_rent_count_zht = df_melted_zht.groupby('Type').agg({'Rent': ['mean', 'count']})

# Filter out types with fewer listings than the threshold
filtered_avg_rent_zht = avg_rent_count_zht[avg_rent_count_zht[('Rent', 'count')] >= MIN_LISTINGS_THRESHOLD_ZHT]

# Sorting the average rent values in descending order
average_rent_sorted_zht = filtered_avg_rent_zht[('Rent', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_zht.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price by Type in the Canton of Zurich')
plt.xlabel('Type')
plt.ylabel('Average Rental Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# City of Zurich
encoded_data_czht = pd.get_dummies(data_czh, columns=['type'])
type_cols_czht = encoded_data_czht.columns[12:]

# Preparing data for the bar chart
df_melted_czht = encoded_data_czht.melt(id_vars=['Rent'], value_vars=type_cols_czht, var_name='Type', value_name='Presence')
df_melted_czht = df_melted_czht[df_melted_czht['Presence'] == 1]

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_CZHT = 3

# Group by type, calculate the average rent and count of listings
avg_rent_count_czht = df_melted_czht.groupby('Type').agg({'Rent': ['mean', 'count']})

# Filter out types with fewer listings than the threshold
filtered_avg_rent_czht = avg_rent_count_czht[avg_rent_count_czht[('Rent', 'count')] >= MIN_LISTINGS_THRESHOLD_CZHT]

# Sorting the average rent values in descending order
average_rent_sorted_czht = filtered_avg_rent_czht[('Rent', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_czht.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price by Type in the City of Zurich')
plt.xlabel('Type')
plt.ylabel('Average Rental Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# (11.) Rental price per Sqm by type
# Canton of Zurich
type_cols_zht = encoded_data_zht.columns[8:]

# Preparing data for the bar chart
df_melted_zht = encoded_data_zht.melt(id_vars=['price_per_sqm'], value_vars=type_cols_zht, var_name='Type', value_name='Presence')
df_melted_zht = df_melted_zht[df_melted_zht['Presence'] == 1]

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_ZHT = 3

# Group by type, calculate the average rent and count of listings
avg_rent_count_zht = df_melted_zht.groupby('Type').agg({'price_per_sqm': ['mean', 'count']})

# Filter out types with fewer listings than the threshold
filtered_avg_rent_zht = avg_rent_count_zht[avg_rent_count_zht[('price_per_sqm', 'count')] >= MIN_LISTINGS_THRESHOLD_ZHT]

# Sorting the average rent values in descending order
average_rent_sorted_zht = filtered_avg_rent_zht[('price_per_sqm', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_zht.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price per sqm by Type in the Canton of Zurich')
plt.xlabel('Type')
plt.ylabel('Average Rental Price per sqm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# City of Zurich
type_cols_czht = encoded_data_czht.columns[12:]

# Preparing data for the bar chart
df_melted_czht = encoded_data_czht.melt(id_vars=['price_per_sqm'], value_vars=type_cols_czht, var_name='Type', value_name='Presence')
df_melted_czht = df_melted_czht[df_melted_czht['Presence'] == 1]

# Define a minimum number of listings threshold
MIN_LISTINGS_THRESHOLD_CZHT = 3

# Group by type, calculate the average rent and count of listings
avg_rent_count_czht = df_melted_czht.groupby('Type').agg({'price_per_sqm': ['mean', 'count']})

# Filter out types with fewer listings than the threshold
filtered_avg_rent_czht = avg_rent_count_czht[avg_rent_count_czht[('price_per_sqm', 'count')] >= MIN_LISTINGS_THRESHOLD_CZHT]

# Sorting the average rent values in descending order
average_rent_sorted_czht = filtered_avg_rent_czht[('price_per_sqm', 'mean')].sort_values(ascending=False)

# Create and show the bar chart
average_rent_sorted_czht.plot(kind='bar', figsize=(10, 7), color='steelblue')
plt.title('Average Rental Price per sqm by Type in the City of Zurich')
plt.xlabel('Type')
plt.ylabel('Average Rental Price per sqm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# (11.) Wordcloud
# Canton of Zurich 
# code to find frequency of each word
def freq(series):
    # Combine all text in the Series into a single string
    combined_str = ' '.join(series)

    # Split the string into a list of words
    words = combined_str.split()
    
    # Initialize a dictionary to count frequency of each word
    freq_dict = {}

    # Count the frequency of each word and store in freq_dict
    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1

    # Create a list of words that are mentioned more than a specified value
    list_words = [word for word, count in freq_dict.items() if count > 400]

    return list_words

# find stopwords
str_series_zh = data_zh['description']
new_stopwords_zh = freq(str_series_zh)
print(f'The new stopwords are: {new_stopwords_zh}')


# adding new stop words to an existing list
extra_stopwords_zh = list(['google','translation','purpose','zurich','chf','hau','no','dich','etc.)',
                        'It','minute','uha','n','ha','wa','st','u','th','andor','etc','questionsread','form',
                        'please','enevread','option','will','included','well','following','using','contact',
                        'final','subject','offer'])
new_words_zh = new_stopwords_zh + extra_stopwords_zh


lemmatizer = WordNetLemmatizer()
#define functuon that removes stopwords from str
def process_text(doc):
    sw = set(new_words_zh)
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', doc)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    output = [word.lower() for word in lem if word.lower() not in sw]
    return ' '.join(output)

#creating variable containing cleaned tweets (no stopwords)
cleaned_text_zh = data_zh['description'].apply(process_text)

# Combine all processed text entries into a single string
combined_text_zh = ' '.join(cleaned_text_zh)

# Create a word cloud
word_cloud_zh = WordCloud(width=1200, height=800, max_words=50, collocations=False).generate(combined_text_zh)

# Display the word cloud using matplotlib
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_zh, interpolation='bilinear')
plt.axis('off')
plt.show()




# City of Zurich
# find stopwords
str_series_czh = data_czh['description']
new_stopwords_czh = freq(str_series_czh)
print(f'The new stopwords are: {new_stopwords_czh}')


# adding new stop words to an existing list
extra_stopwords_czh = list(['google','translation','purpose','zurich','chf','hau','no','dich','etc.)',
                        'It','minute','uha','n','ha','wa','st','u','th','andor','etc','questionsread','form',
                        'please','enevread','option','will','included','well','following','using','contact',
                        'final','subject','offer'])
new_words_czh = new_stopwords_czh + extra_stopwords_czh


lemmatizer = WordNetLemmatizer()
#define functuon that removes stopwords from str
def process_text(doc):
    sw = set(new_words_czh)
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', doc)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    output = [word.lower() for word in lem if word.lower() not in sw]
    return ' '.join(output)

#creating variable containing cleaned tweets (no stopwords)
cleaned_text_czh = data_czh['description'].apply(process_text)

# Combine all processed text entries into a single string
combined_text_czh = ' '.join(cleaned_text_czh)

# Create a word cloud
word_cloud_czh = WordCloud(width=1200, height=800, max_words=50, collocations=False).generate(combined_text_czh)

# Display the word cloud using matplotlib
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_czh, interpolation='bilinear')
plt.axis('off')
plt.show()



###############################################################################

# Data Preparation for Model Prediction (Canton of Zurich - Rental price)

###############################################################################

# ensure that both type and City/District are transformed
encoded_data_zh = pd.get_dummies(data_zh, columns=['type','City'])
encoded_data_czh = pd.get_dummies(data_czh, columns=['type','District'])

# remove price_per_sqm from the dataset to prevent multicollinearity
encoded_data_zh.drop('price_per_sqm', axis=1, inplace=True)
encoded_data_zh.to_csv('Immobilier.ch_database_ZH_7_cov_encoded.csv', index=False)

# Split data into train and test sets
np.random.seed(11052022)
train, test = train_test_split(encoded_data_zh, test_size=0.2, random_state=11052022)
print(train.describe())
print(test.describe())


# Define variables
Y_NAME = 'Rent'
X_NAME = list(encoded_data_zh.columns[2:])

###############################################################################

# Model Prediction (Canton of Zurich)


### 1.) Hedonic Pricing Model
hdn_model, hdn_model_summary, hdn_predictions = functions.hedonic_pricing_model(train, test, Y_NAME, X_NAME)



### 2.) Regularization Regression Model (Lasso)
lasso_model, lasso_best_alpha, lasso_coefficients, lasso_predictions = functions.lasso_regression(train, test, Y_NAME, X_NAME, nfolds=10)



### 3.) Regularization Regression Model (Ridge)
ridge_model, ridge_best_alpha, ridge_coefficients, ridge_predictions = functions.ridge_regression(train, test, Y_NAME, X_NAME, nfolds=10)      



### 4.) Decision Trees
tree_model, tree_predictions = functions.decision_tree(train, test, Y_NAME, X_NAME)



### 5.) Random forest (RF)
rf_model, rf_predictions = functions.random_forest(train, test, Y_NAME, X_NAME, n_trees=10)



### 6.) Gradient Boosting Models (i.e., XGBoost)
xgb_model, xgb_predictions = functions.xgboost(train[X_NAME], train[Y_NAME], 
                                               test[X_NAME], test[Y_NAME])


### 7.) Support Vector Regression (SVR)
svr_model, svr_predictions = functions.svr(train, test, Y_NAME, X_NAME)


### 8.) Neural Network
nn_model, nn_evaluation, nn_predictions = functions.neural_network(
    train_X=train[X_NAME], 
    train_Y=train[Y_NAME], 
    test_X=test[X_NAME], 
    test_Y=test[Y_NAME],
    input_dim=83
)


###############################################################################

# Model Evaluation (Out-of-sample MSE and R-Square)


# Calculate Out-of-sample MSE
hedonic_mse = mean_squared_error(test[Y_NAME], hdn_predictions)
lasso_mse = mean_squared_error(test[Y_NAME], lasso_predictions)
ridge_mse = mean_squared_error(test[Y_NAME], ridge_predictions)
tree_mse = mean_squared_error(test[Y_NAME], tree_predictions)
rf_mse = mean_squared_error(test[Y_NAME], rf_predictions)
xgb_mse = mean_squared_error(test[Y_NAME], xgb_predictions)
svr_mse = mean_squared_error(test[Y_NAME], svr_predictions)
nn_mse = mean_squared_error(test[Y_NAME], nn_predictions)




# Calculate R-Squared
rsq_hedonic = functions.calculate_r_squared(test[Y_NAME], hdn_predictions)
rsq_lasso = functions.calculate_r_squared(test[Y_NAME], lasso_predictions)
rsq_ridge = functions.calculate_r_squared(test[Y_NAME], ridge_predictions)
rsq_tree = functions.calculate_r_squared(test[Y_NAME], tree_predictions)
rsq_rf = functions.calculate_r_squared(test[Y_NAME], rf_predictions)
rsq_xgb = functions.calculate_r_squared(test[Y_NAME], xgb_predictions)
rsq_svr = functions.calculate_r_squared(test[Y_NAME], svr_predictions)
nn_predictions_flat = nn_predictions[:, 0] # Flatten the predictions for Neural Network
rsq_nn = functions.calculate_r_squared(test[Y_NAME], nn_predictions_flat)



# Model Evaluation
model_evaluation = pd.DataFrame({
    'Model': ['Hedonic Pricing model', 'Lasso model', 'Ridge model', 'Decision Tree model', 
              'Random Forest model', 'XGBoost model', 'SVR model', 'Neural Network model'],
    'Out_of_sample_MSE': [hedonic_mse, lasso_mse, ridge_mse, tree_mse, rf_mse, xgb_mse, svr_mse, nn_mse],
    'R_squared': [rsq_hedonic, rsq_lasso, rsq_ridge, rsq_tree, rsq_rf, rsq_xgb, rsq_svr, rsq_nn]
})

# Print the DataFrame
print(model_evaluation)



###############################################################################

# Variable Importance

# (1.) XGBoost

# Variable Importance Plot (full)
fig, ax = plt.subplots(figsize=(20, 25))
xgb.plot_importance(xgb_model, ax=ax)
plt.title('Feature Importance')
plt.show()

# Variable Importance Plot (top 15)
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xgb_model, ax=ax, max_num_features=15)
plt.title('Top 15 XGBoost Feature Importance for Canton of Zurich')
plt.show()

# Print XGBoost score
import xgboost as xgb

feature_importances = xgb_model.get_score()
sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

# Print the sorted feature importances
for feature, importance in sorted_feature_importances:
    print(f"{feature}: {importance}")



# (2.) Decision Tree
feature_importances_dt = tree_model.feature_importances_

# Pair feature names with their importance scores
importance_tree = sorted(zip(X_NAME, feature_importances_dt), key=lambda x: x[1], reverse=True)

# Display the feature importance
for name, importance in importance_tree:
    print(f"{name}: {importance}")
    
# Plot for Decision Tree Feature Importances
top_15_importance_tree = importance_tree[:15]
plt.figure(figsize=(20, 10))
plt.barh([name for name, _ in reversed(top_15_importance_tree)], 
         [importance for _, importance in reversed(top_15_importance_tree)])
plt.title('Top 15 Decision Tree Feature Importances for Canton of Zurich')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# (3.) Random Forest
feature_importances_rf = rf_model.feature_importances_

# Pair feature names with their importance scores
importance_rf = sorted(zip(X_NAME, feature_importances_rf), key=lambda x: x[1], reverse=True)

# Display the feature importance
for name, importance in importance_rf:
    print(f"{name}: {importance}")

# Plot for Decision Tree Feature Importances
top_15_importance_rf = importance_rf[:15]
plt.figure(figsize=(20, 10))
plt.barh([name for name, _ in reversed(top_15_importance_rf)], 
         [importance for _, importance in reversed(top_15_importance_rf)])
plt.title('Top 15 Random Forest Feature Importances for Canton of Zurich')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# (4.) Lasso
importance_lasso = lasso_coefficients.abs().sort_values(ascending=False)

print("Variable Importance from Lasso Regression:")
print(importance_lasso)

# Display the feature importance
top_15_importance_lasso = importance_lasso.nlargest(15)
plt.figure(figsize=(20, 10))
plt.barh(y=top_15_importance_lasso.index[::-1], width=top_15_importance_lasso.values[::-1])
plt.title('Top 15 Lasso Regression Feature Importances for Canton of Zurich')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.show()


###############################################################################

# Data Preparation for Model Prediction (City of Zurich - Rental price)

###############################################################################

# remove price_per_sqm from the dataset to prevent multicollinearity
encoded_data_czh.drop('price_per_sqm', axis=1, inplace=True)
encoded_data_czh.to_csv('Immobilier.ch_database_ZH_9_cov_encoded.csv', index=False)

# Split data into train and test sets
np.random.seed(11052022)
train_czh, test_czh = train_test_split(encoded_data_czh, test_size=0.2, random_state=11052022)
print(train_czh.describe())
print(test_czh.describe())


# Define variables
Y_NAME_czh = 'Rent'
X_NAME_czh = list(encoded_data_czh.columns[2:4]) + list(encoded_data_czh.columns[5:])

###############################################################################

# Model Prediction (City of Zurich)


### 1.) Hedonic Pricing Model
hdn_model_czh, hdn_model_summary_czh, hdn_predictions_czh = functions.hedonic_pricing_model(train_czh, test_czh, Y_NAME_czh, X_NAME_czh)



### 2.) Regularization Regression Model (Lasso)
lasso_model_czh, lasso_best_alpha_czh, lasso_coefficients_czh, lasso_predictions_czh = functions.lasso_regression(train_czh, test_czh, Y_NAME_czh, X_NAME_czh, nfolds=10)



### 3.) Regularization Regression Model (Ridge)
ridge_model_czh, ridge_best_alpha_czh, ridge_coefficients_czh, ridge_predictions_czh = functions.ridge_regression(train_czh, test_czh, Y_NAME_czh, X_NAME_czh, nfolds=10)      



### 4.) Decision Trees
tree_model_czh, tree_predictions_czh = functions.decision_tree(train_czh, test_czh, Y_NAME_czh, X_NAME_czh)



### 5.) Random forest (RF)
rf_model_czh, rf_predictions_czh = functions.random_forest(train_czh, test_czh, Y_NAME_czh, X_NAME_czh, n_trees=10)



### 6.) Gradient Boosting Models (i.e., XGBoost)
xgb_model_czh, xgb_predictions_czh = functions.xgboost(train_czh[X_NAME_czh], train_czh[Y_NAME_czh], 
                                               test_czh[X_NAME_czh], test_czh[Y_NAME_czh])


### 7.) Support Vector Regression (SVR)
svr_model_czh, svr_predictions_czh = functions.svr(train_czh, test_czh, Y_NAME_czh, X_NAME_czh)



### 8.) Neural Network
nn_model_czh, nn_evaluation_czh, nn_predictions_czh = functions.neural_network(
    train_X=train_czh[X_NAME_czh], 
    train_Y=train_czh[Y_NAME_czh], 
    test_X=test_czh[X_NAME_czh], 
    test_Y=test_czh[Y_NAME_czh],
    input_dim=43
)


###############################################################################

# Model Evaluation (Out-of-sample MSE and R-Square)


# Calculate Out-of-sample MSE
hedonic_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], hdn_predictions_czh)
lasso_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], lasso_predictions_czh)
ridge_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], ridge_predictions_czh)
tree_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], tree_predictions_czh)
rf_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], rf_predictions_czh)
xgb_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], xgb_predictions_czh)
svr_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], svr_predictions_czh)
nn_mse_czh = mean_squared_error(test_czh[Y_NAME_czh], nn_predictions_czh)




# Calculate R-Squared
rsq_hedonic_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], hdn_predictions_czh)
rsq_lasso_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], lasso_predictions_czh)
rsq_ridge_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], ridge_predictions_czh)
rsq_tree_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], tree_predictions_czh)
rsq_rf_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], rf_predictions_czh)
rsq_xgb_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], xgb_predictions_czh)
rsq_svr_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], svr_predictions_czh)
nn_predictions_flat_czh = nn_predictions_czh[:, 0] # Flatten the predictions for Neural Network
rsq_nn_czh = functions.calculate_r_squared(test_czh[Y_NAME_czh], nn_predictions_flat_czh)



# Model Evaluation
model_evaluation_czh = pd.DataFrame({
    'Model': ['Hedonic Pricing model', 'Lasso model', 'Ridge model', 'Decision Tree model', 
              'Random Forest model', 'XGBoost model', 'SVR model', 'Neural Network model'],
    'Out_of_sample_MSE': [hedonic_mse_czh, lasso_mse_czh, ridge_mse_czh, tree_mse_czh, rf_mse_czh, xgb_mse_czh, svr_mse_czh, nn_mse_czh],
    'R_squared': [rsq_hedonic_czh, rsq_lasso_czh, rsq_ridge_czh, rsq_tree_czh, rsq_rf_czh, rsq_xgb_czh, rsq_svr_czh, rsq_nn_czh]
})

# Print the DataFrame
print(model_evaluation_czh)


###############################################################################

# Variable Importance

# (1.) XGBoost

# Variable Importance Plot (full)
fig, ax = plt.subplots(figsize=(20, 25))
xgb.plot_importance(xgb_model_czh, ax=ax)
plt.title('Feature Importance')
plt.show()

# Variable Importance Plot (top 15)
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xgb_model_czh, ax=ax, max_num_features=15)
plt.title('Top 15 XGBoost Feature Importance for City of Zurich')
plt.show()

# Print XGBoost score
feature_importances_czh = xgb_model_czh.get_score()
sorted_feature_importances_czh = sorted(feature_importances_czh.items(), key=lambda x: x[1], reverse=True)

# Print the sorted feature importances
for feature, importance in sorted_feature_importances_czh:
    print(f"{feature}: {importance}")




# (2.) Decision Tree
feature_importances_dt_czh = tree_model_czh.feature_importances_

# Pair feature names with their importance scores
importance_tree_czh = sorted(zip(X_NAME_czh, feature_importances_dt_czh), key=lambda x: x[1], reverse=True)

# Display the feature importance
for name, importance in importance_tree_czh:
    print(f"{name}: {importance}")
    
# Plot for Decision Tree Feature Importances
top_15_importance_tree_czh = importance_tree_czh[:15]
plt.figure(figsize=(20, 10))
plt.barh([name for name, _ in reversed(top_15_importance_tree_czh)], 
         [importance for _, importance in reversed(top_15_importance_tree_czh)])
plt.title('Top 15 Decision Tree Feature Importances for City of Zurich')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# (3.) Random Forest
feature_importances_rf_czh = rf_model_czh.feature_importances_

# Pair feature names with their importance scores
importance_rf_czh = sorted(zip(X_NAME_czh, feature_importances_rf), key=lambda x: x[1], reverse=True)

# Display the feature importance
for name, importance in importance_rf_czh:
    print(f"{name}: {importance}")

# Plot for Decision Tree Feature Importances
top_15_importance_rf_czh = importance_rf_czh[:15]
plt.figure(figsize=(20, 10))
plt.barh([name for name, _ in reversed(top_15_importance_rf_czh)], 
         [importance for _, importance in reversed(top_15_importance_rf_czh)])
plt.title('Top 15 Random Forest Feature Importances for City of Zurich')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# (4.) Lasso
importance_lasso_czh = lasso_coefficients_czh.abs().sort_values(ascending=False)

print("Variable Importance from Lasso Regression:")
print(importance_lasso_czh)

# Display the feature importance
top_15_importance_lasso_czh = importance_lasso_czh.nlargest(15)
plt.figure(figsize=(20, 10))
plt.barh(y=top_15_importance_lasso_czh.index[::-1], width=top_15_importance_lasso_czh.values[::-1])
plt.title('Top 15 Lasso Regression Feature Importances for City of Zurich')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.show()



