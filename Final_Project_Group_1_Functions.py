#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Data Analytics

Fall Semester 2023.

University of St. Gallen.

Jidapa Lengthaisong (22-601-355) and Quentin Leoni (19-406-560)

"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
import re




# set pandas printing option to see all columns of the data
pd.set_option('display.max_columns', 100)
# and without breaking the dataframe into several lines
pd.set_option('expand_frame_repr', False)
# justify the headers for pandas dataframes
pd.set_option('display.colheader_justify', 'right')


# define an Output class for simultaneous console - file output
class Output():
    """Output class for simultaneous console/file output."""

    def __init__(self, path, name):

        self.terminal = sys.stdout
        self.output = open(path + '/' + name + ".txt", "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""


def clean_rent_price_and_living_space(df, price_column, space_column):
    """
    Cleans the rent price and living space columns by extracting numeric values.
    """
    df[price_column] = df[price_column].str.extract('(\d+,?\d*)')[0]
    df[space_column] = df[space_column].str.extract('(\d+)')[0]

    # Convert to numeric
    df[price_column] = pd.to_numeric(df[price_column].str.replace(',', ''), errors='coerce')
    df[space_column] = pd.to_numeric(df[space_column], errors='coerce')
    return df

def remove_unnecessary_columns(df, columns_to_remove):
    """
    Removes unnecessary columns from the dataframe.
    """
    return df.drop(columns=columns_to_remove)

def clean_year_column(data):
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year'])
    data = data[data['year']>1000]
    data.dropna(inplace=True)
    return data

def clean_livingspace_column(data):
    data['living space'] = pd.to_numeric(data['living space'], errors='coerce')
    data = data.dropna(subset=['living space'])
    data = data[data['living space'] < 1000]
    return data

def clean_data(data, price_column, space_column, drop_nan_columns=False, fill_nan=None, drop_nan_rows=False):
    """
    Loads, cleans, and returns the cleaned dataframe.
    """

    # Clean rent price and living space columns
    df = clean_rent_price_and_living_space(data, price_column, space_column)

    # Drop rows where most columns are NaN
    df = df.dropna(how='all')

    # Drop columns where all values are NaN
    drop_nan_columns = True
    if drop_nan_columns:
        df.dropna(axis=1, how='all', inplace=True)
        
    # Drop rows with any NaN values
    df.dropna(inplace=True)

    return df


def clean_text_column(column):
    # Convert to string
    column = column.astype(str)

    # Basic cleaning: lowercasing, stripping whitespace
    column = column.str.lower().str.strip()

    return column



# Function to clean the 'features' column
def clean_features(feature_string):
    # Remove quotes, square brackets, and extra spaces
    feature_string = feature_string.strip("[]' ")

    # Split the features into a list based on ',' and '\n'
    features = re.split(r',\s*|\n', feature_string)

    # Clean and standardize each feature
    cleaned_features = [feature.strip().lower() for feature in features if feature.strip()]
    
    if pd.isna(feature_string) or feature_string == '':
        return "no features"

    # Join the cleaned features back into a single string
    return ', '.join(cleaned_features)



# write a function to produce summary stats:
# mean, variance, standard deviation, maximum and minimum,
# the number of missings, unique values and number of observations
def my_summary_stats(data):
    """
    Summary stats: mean, variance, standard deviation, maximum and minimum.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe for which descriptives will be computed
    Returns

    -------
    None. Prints descriptive table od the data
    """
    # generate storage for the stats as an empty dictionary
    my_descriptives = {}
    # loop over columns
    for col_id in data.columns:
        # fill in the dictionary with descriptive values by assigning the
        # column ids as keys for the dictionary
        my_descriptives[col_id] = [data[col_id].mean(),                  # mean
                                   data[col_id].var(),               # variance
                                   data[col_id].std(),                # st.dev.
                                   data[col_id].max(),                # maximum
                                   data[col_id].min(),                # minimum
                                   sum(data[col_id].isna()),          # missing
                                   len(data[col_id].unique()),  # unique values
                                   data[col_id].shape[0]]      # number of obs.
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=['mean', 'var', 'std', 'max', 'min',
                                          'na', 'unique', 'obs']).transpose()
    # define na, unique and obs as integers such that no decimals get printed
    ints = ['na', 'unique', 'obs']
    # use the .astype() method of pandas dataframes to change the type
    my_descriptives[ints] = my_descriptives[ints].astype(int)
    # print the descriptives, (\n inserts a line break)
    print('Descriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, '\n\n', sep='\n')
    
    
    
    
# write a function to produce correlation heatmap
def correlation_heatmap(data):
    """"
    Create correlation heatmap

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing variables of interest

    Returns
    -------
    Prints the correlation heatmap
    """
    # Select only numeric columns
    data_numeric = data.select_dtypes(include='number')

    # Compute the correlation matrix
    corr_matrix = data_numeric.corr()

    # Initialize the matplotlib figure
    plot.figure(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'label': 'Correlation'})

    # Customizations for aesthetics
    plot.title('Correlation Heatmap')
    plot.xticks(rotation=90)
    plot.yticks(rotation=0)
    plot.show()




# write a function to produce hedonic pricing model
def hedonic_pricing_model(train_data, test_data, dependent_var, independent_vars):
    """
    Fit a Hedonic Pricing Model (Linear Regression) and make predictions.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training dataset.
    test_data : pd.DataFrame
        DataFrame containing the test dataset.
    dependent_var : str
        The name of the outcome variable (dependent variable).
    independent_vars : list
        List of names of the independent variables.

    Returns
    -------
    model_summary : str
        Summary of the fitted regression model.
    predictions : pd.Series
        Predictions made on the test dataset.
    """
    # Prepare the training data
    X_train = train_data[independent_vars]
    y_train = train_data[dependent_var]

    # Add a constant to the model (for the intercept)
    X_train = sm.add_constant(X_train)

    # Fit the linear regression model
    model = sm.OLS(y_train, X_train).fit()
    model_summary = model.summary()

    # Prepare the test data
    X_test = test_data[independent_vars]
    X_test = sm.add_constant(X_test)  # Add constant for the test set

    # Make predictions on test set
    predictions = model.predict(X_test)

    return model, model_summary, predictions





# write a function to produce lasso regression
def lasso_regression(train_data, test_data, dependent_var, independent_vars, nfolds=10):
    """
    Fit a cross-validated Lasso Regression model and make predictions.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training dataset.
    test_data : pd.DataFrame
        DataFrame containing the test dataset.
    dependent_var : str
        The name of the outcome variable (dependent variable).
    independent_vars : list
        List of names of the independent variables.
    nfolds : int
        Number of folds for cross-validation.

    Returns
    -------
    best_alpha : float
        The best alpha value determined by cross-validation.
    predictions : np.array
        Predictions made on the test dataset.
    coefficients : pd.Series
        Coefficients of the model.
    """
    # Preparing the training data
    X_train = train_data[independent_vars]
    Y_train = train_data[dependent_var]

    # Standardizing the features (important for Lasso)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and fit the LassoCV model
    lasso_cv = LassoCV(alphas=None, cv=nfolds, max_iter=100000, normalize=False)
    lasso_cv.fit(X_train_scaled, Y_train)

    # The best alpha value
    best_alpha = lasso_cv.alpha_

    # Preparing the test data
    X_test = test_data[independent_vars]
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler as for training

    # Make predictions on the test set using the best alpha
    lasso_model = Lasso(alpha=best_alpha)
    lasso_model.fit(X_train_scaled, Y_train)  # Refit using the best alpha
    predictions = lasso_model.predict(X_test_scaled)

    # Coefficients of the model
    coefficients = pd.Series(lasso_model.coef_, index=independent_vars)
    
    # Print results
    print("Best alpha (Regularization Parameter):", best_alpha)
    print("\nCoefficients of the Lasso Model:")
    print(coefficients)

    return lasso_model, best_alpha, coefficients, predictions




# write a function to produce ridge regression
def ridge_regression(train_data, test_data, dependent_var, independent_vars, nfolds=10):
    """
    Fit a cross-validated Ridge Regression model and make predictions.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training dataset.
    test_data : pd.DataFrame
        DataFrame containing the test dataset.
    dependent_var : str
        The name of the outcome variable (dependent variable).
    independent_vars : list
        List of names of the independent variables.
    nfolds : int
        Number of folds for cross-validation.

    Returns
    -------
    best_alpha : float
        The best alpha value determined by cross-validation.
    predictions : np.array
        Predictions made on the test dataset.
    coefficients : pd.Series
        Coefficients of the model.
    """
    # Preparing the training data
    X_train = train_data[independent_vars]
    Y_train = train_data[dependent_var]

    # Standardizing the features (important for Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and fit the RidgeCV model
    ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=nfolds, normalize=False)
    ridge_cv.fit(X_train_scaled, Y_train)

    # The best alpha value
    best_alpha = ridge_cv.alpha_

    # Preparing the test data
    X_test = test_data[independent_vars]
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler as for training

    # Make predictions on the test set using the best alpha
    ridge_model = RidgeCV(alphas=[best_alpha], normalize=False)
    ridge_model.fit(X_train_scaled, Y_train)  # Refit using the best alpha
    predictions = ridge_model.predict(X_test_scaled)

    # Coefficients of the model
    coefficients = pd.Series(ridge_model.coef_, index=independent_vars)
    
    # Print results
    print("Best alpha (Regularization Parameter):", best_alpha)
    print("\nCoefficients of the Ridge Model:")
    print(coefficients)

    return ridge_model, best_alpha, coefficients, predictions





# write a function to produce decision tree
def decision_tree(train_data, test_data, dependent_var, independent_vars):
    """
    Fit a Decision Tree Regression model and make predictions.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training dataset.
    test_data : pd.DataFrame
        DataFrame containing the test dataset.
    dependent_var : str
        The name of the outcome variable (dependent variable).
    independent_vars : list
        List of names of the independent variables.

    Returns
    -------
    tree_model : sklearn.tree.DecisionTreeRegressor
        The fitted Decision Tree model.
    predictions : np.array
        Predictions made on the test dataset.
    """
    # Preparing the data
    X_train = train_data[independent_vars]
    Y_train = train_data[dependent_var]
    X_test = test_data[independent_vars]

    # Initialize and fit the Decision Tree model
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X_train, Y_train)

    # Make predictions on the test set
    predictions = tree_model.predict(X_test)

    # Plotting the tree
    plot.figure(figsize=(20,10))
    plot_tree(tree_model, feature_names=independent_vars, filled=True)
    plot.show()

    return tree_model, predictions





# write a function to produce random forest
def random_forest(train_data, test_data, dependent_var, independent_vars, n_trees=50):
    """
    Fit a Random Forest Regression model and make predictions.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training dataset.
    test_data : pd.DataFrame
        DataFrame containing the test dataset.
    dependent_var : str
        The name of the outcome variable (dependent variable).
    independent_vars : list
        List of names of the independent variables.
    n_trees : int
        Number of trees in the forest.

    Returns
    -------
    rf_model : sklearn.ensemble.RandomForestRegressor
        The fitted Random Forest model.
    predictions : np.array
        Predictions made on the test dataset.
    """
    # Preparing the data
    X_train = train_data[independent_vars]
    Y_train = train_data[dependent_var]
    X_test = test_data[independent_vars]

    # Initialize and fit the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=n_trees)
    rf_model.fit(X_train, Y_train)

    # Make predictions on the test set
    predictions = rf_model.predict(X_test)

    # Print the model summary
    print(rf_model)

    return rf_model, predictions




# write a function to produce XGBoost
def xgboost(train_X, train_Y, test_X, test_Y, num_rounds=100, early_stopping_rounds=10):
    """
    Fit an XGBoost Regression model and make predictions.

    Parameters
    ----------
    train_X : DataFrame or array-like
        Features of the training dataset.
    train_y : Series or array-like
        Target variable of the training dataset.
    test_X : DataFrame or array-like
        Features of the test dataset.
    test_y : Series or array-like
        Target variable of the test dataset.
    num_rounds : int, optional
        Number of boosting rounds.
    early_stopping_rounds : int, optional
        Validation metric needs to improve at least once in every
        early_stopping_rounds rounds to continue training.

    Returns
    -------
    model : xgboost.core.Booster
        The trained XGBoost model.
    predictions : array
        Predictions made on the test dataset.
    """
    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X, label=test_Y)

    params = {
        "objective": "reg:squarederror",
        "eta": 0.3,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1,
        "nthread": 2,
        "eval_metric": "rmse"
    }

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_rounds, 
                      evals=watchlist, early_stopping_rounds=early_stopping_rounds, 
                      verbose_eval=10)

    predictions = model.predict(dtest)
    return model, predictions





# write a function to produce SVR
def svr(train_data, test_data, dependent_var, independent_vars):
    """
    Fit an SVM Regression model and make predictions.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training dataset.
    test_data : pd.DataFrame
        DataFrame containing the test dataset.
    dependent_var : str
        The name of the outcome variable (dependent variable).
    independent_vars : list
        List of names of the independent variables.

    Returns
    -------
    svm_model : sklearn.svm.SVR
        The fitted SVM model.
    predictions : np.array
        Predictions made on the test dataset.
    """
    # Preparing the data
    X_train = train_data[independent_vars]
    Y_train = train_data[dependent_var]
    X_test = test_data[independent_vars]

    # Initialize and fit the SVM model
    svm_model = SVR()
    svm_model.fit(X_train, Y_train)

    # Make predictions on the test set
    predictions = svm_model.predict(X_test)

    return svm_model, predictions




# write a function to produce Neural Network
def neural_network(train_X, train_Y, test_X, test_Y, input_dim, epochs=50, batch_size=10):
    """
    Fit a Neural Network for regression and evaluate on test data.

    Parameters
    ----------
    train_X : array-like
        Features of the training dataset.
    train_Y : array-like
        Target variable of the training dataset.
    test_X : array-like
        Features of the test dataset.
    test_Y : array-like
        Target variable of the test dataset.
    input_dim : int
        Number of input variables (features).
    epochs : int
        Number of epochs to train the model.
    batch_size : int
        Number of samples per gradient update.

    Returns
    -------
    model : keras.engine.sequential.Sequential
        The trained neural network model.
    evaluation : dict
        Loss and accuracy of the model on the test dataset.
    predictions : array
        Predictions made by the model on the test dataset.
    """
    # Define the model
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # Fit the model
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=1)

    # Evaluate the model
    evaluation = model.evaluate(test_X, test_Y, verbose=0)

    # Make predictions
    predictions = model.predict(test_X)

    return model, evaluation, predictions



# write a function to calculate R-Squared
def calculate_r_squared(test_Y, predictions):
    sst = ((test_Y - test_Y.mean()) ** 2).sum()
    ssr = ((test_Y - predictions) ** 2).sum()
    r_squared = 1 - (ssr / sst)
    return r_squared











