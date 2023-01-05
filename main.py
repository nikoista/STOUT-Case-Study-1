#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
@author: nikoista
"""
 
# Basic Numpy , Pandas , Matplotlib Libraries
import numpy as np
import pandas as pd
import warnings
 
 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 
from mrmr import mrmr_regression
#from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
 



# PUT YOUR FILE PATH HERE
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
path = 'C:/Users/__________'
dataset_path = 'C:/Users/__________'
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
 





############################ Performance Metrics ##############################
def performance_Metrics(y, y_pred,X):
###################### Calculating RMSE,SStot,SSres Score #####################
    MSE=0    
    ss_tot = 0    
    ss_res = 0
    mean_y = np.mean(y)
 
    for i in range(0,len(y)):
         MSE   += ( (y[i] - y_pred[i]) ** 2)
         ss_tot += ((y[i] - mean_y) ** 2)
         ss_res += ((y[i] - y_pred[i]) ** 2)
        
########################### AdjR^2 and NRMSE Metrics ##########################
    RMSE = np.sqrt(MSE/len(y))
    NRMSE = (RMSE/(np.std(y)))
    
    R2 = 1 - (ss_res/ss_tot)
    p = len(X[0])
    N = len(X)
    adjR2 = (1 - (((1-R2)*(N-1))/(N-p-1)))
    
########################## Residuals and Standard Residuals ###################
    residuals = (y - y_pred)
    var_residuals=np.sum (y - y_pred)*np.sum (y - y_pred)/(len(y)-2)
    standar_residuals = (residuals/var_residuals)
    
    return  adjR2 , MSE  ,NRMSE,standar_residuals
###############################################################################


   
     
###############################################################################

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("\n Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
###############################################################################
 


#########################  Information About our Data #########################
def data_Info (data):
    data.info()
    data.head()
    # Statistics for each column
    data.describe()
    feature_names = [i for i in data.columns ]
    #print('Statistical View of our Data: \n' ,data.describe().T)
    return feature_names
###############################################################################

def explanatory_Analysis_Visualization(reduced_df):
     
     # Create the default pairplot
     sns.pairplot(reduced_df)
     plt.show()
     
     # Create the heatmap
     corr = reduced_df.corr()
     sns.heatmap(corr, cmap="YlGnBu")
     plt.show()
     
     
###############################################################################
def data_5_Visualization(df):
    
   plt.figure(figsize=(20,15))
   # Get the count of each unique loan purpose
   loan_purpose_counts = df['loan_purpose'].value_counts()

   # Create the bar chart
   sns.barplot(x=loan_purpose_counts.index, y=loan_purpose_counts.values)

   # Set the x-axis labels and y-axis labels
   plt.xlabel('Loan Purpose')
   plt.ylabel('Count')

   # Set the title of the plot
   plt.title('Distribution of Loan Purposes')

   # Show the plot
   plt.show()  
   
   plt.figure(figsize=(10,10))
   # Create the scatter plot
   plt.scatter(df['loan_amount'], df['interest_rate'])

   # Add x and y labels
   plt.xlabel('Loan Amount')
   plt.ylabel('Interest Rate')

   # Show the plot
   plt.show()
   
   plt.figure(figsize=(10,10))
   # Create a box plot
   sns.boxplot(df["debt_to_income"])

   # Show the plot
   plt.show()
   #This will create a box plot showing the distribution of debt-to-income ratios in the data. The box itself represents the interquartile range (IQR) of the data, with the line in the middle of the box representing the median. The whiskers extend to the minimum and maximum values, and any points outside the whiskers are considered outliers.
   
   
   plt.figure(figsize=(15,10))
   df['loan_id'] = range(1, len(df) + 1)
   # Extract the month and year of the loan issuance
   df['issue_date'] = pd.to_datetime(clean_data['issue_month'], format='%b-%Y')

   # Group the data by year and month, and count the number of loans issued in each period
   loans_by_time = df.groupby(['issue_date'])['loan_id'].count()

   # Plot the data
   plt.plot(loans_by_time)
   plt.xlabel('Issue Date')
   plt.ylabel('Number of Loans Issued')
   plt.title('Trend of Loan Volumes Over Time')
   plt.show()
   
   
   plt.figure(figsize=(10,10))
   sns.distplot(df['loan_amount'], hist=True, kde=False, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})

   # Add a title and labels to the plot
   plt.title("Distribution of Loan Amounts")
   plt.xlabel("Loan Amount")
   plt.ylabel("Number of Loans")

    
###############################################################################
 

###############################################################################
def results_Visualization(y_test,rfr_y_pred,linear_y_pred,linear_resids,rfr_resids):
    
    # Set the figure size
    plt.figure(figsize=(20, 20))

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
   
    # Create the first plot
    ax1.scatter(linear_y_pred,linear_resids)

    # Create the second plot
    ax2.scatter(rfr_y_pred,rfr_resids)
    
    # Add a title and labels
    plt.suptitle('Linear Regression - Random Forest Regression')
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Standardized Residuals')
    ax2.set_xlabel('Predictions')
    ax2.set_ylabel('Standardized Residuals')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.2)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.2)

    #Add a legend
    plt.legend()
 
    # Show the plot
    plt.show()
    
    
    
    # Set the figure size
    plt.figure(figsize=(20, 20))

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
    
    # Create the first plot
    ax1.scatter(y_test,linear_y_pred,c='crimson')
    p1 = max(max(linear_y_pred), max(y_test))
    p2 = min(min(linear_y_pred), min(y_test))
    ax1.plot([p1, p2], [p1, p2], 'b-',lw=4)

    # Create the second plot
    ax2.scatter(y_test,rfr_y_pred,c='crimson')
    p1 = max(max(rfr_y_pred), max(y_test))
    p2 = min(min(rfr_y_pred), min(y_test))
    ax2.plot([p1, p2], [p1, p2], 'b-',lw=4)
    
    # Add a title and labels
    plt.suptitle('Linear Regression - Random Forest Regression')
    ax1.set_xlabel('True Values', fontsize=15)
    ax1.set_ylabel('Predictions', fontsize=15)
    ax2.set_xlabel('True Values', fontsize=15)
    ax2.set_ylabel('Predictions', fontsize=15)
 
    # Show the plot
    plt.show()
 
 
 ###############################################################################
    
def row_cols_df(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    
    return cols,num_cols,cat_cols






########################## Entry point for the program ########################
if __name__ == '__main__':
    
########################### Import Real Dataset ################################
     data = pd.read_csv(dataset_path+ 'loans_full_schema.csv') #sep=';')
     feature_names = data_Info(data)
     df = data.copy()
     cols,num_cols,cat_cols =  row_cols_df(df)
      
     
######################### DATA PREPROCESSING AND CLEANING #####################

     # Remove duplicates
     data.drop_duplicates(inplace=True)
    
     # Begin the cleaning process with removing missing values colums > 50%
     missing =  missing_values_table(data)
     print(missing)
     missing_names =  [i for i in missing.index ]
     # Drop the columns with > 50% missing
     for i in range(0,len(missing)):
         if missing.iloc[i,0]*100/10000>75:
             print('\n Missing Values Columns Remover ',i)
             data.drop(columns = missing_names[i], axis=1, inplace=True)
 
     # replace missing values of columns with missing values < 50%
     print('\n')
     missing =  missing_values_table(data)

     cols,num_cols,cat_cols =  row_cols_df(data)
  
     # Drop it cause it has only 0.0 values
     print(data.describe())
 
     # Remove Columns that have only one value (i.e 0.0)
     data=data.drop(columns=['num_accounts_120d_past_due'])
 
     missing =  missing_values_table(data)
     cols,num_cols,cat_cols =  row_cols_df(data)
     #data = pd.concat((data[num_cols],data[cat_cols]),axis=1)
     
     
     
     # Impute missing continious values with the columns mean for missing <20%
     where = data[num_cols].isnull().sum()>0
     where = num_cols[where]
     check = [col  for col, dt in data[missing.index].dtypes.items() if dt == object]
     for i in range(0,len(missing)):
         if missing.iloc[i,0]*100/10000<20:
             for j in range(0,len(check)):
                 if  check[j] != missing.index[i]:
                     mean = data[missing.index[i]].mean()
                     data[missing.index[i]].fillna(mean, inplace=True)
 
     missing =  missing_values_table(data)
     print(missing)
     missing_data = data['months_since_last_delinq']
     data=data.drop(columns=['months_since_last_delinq'])
     # Impute continious values with the prediction
     
     
     # Impute catecorical values with the most frequent category
     # Calculate the frequency of each category
     freq = data['emp_title'].value_counts()

     # Select the most frequent category
     most_frequent = freq.index[0]

     # Impute the missing values with the most frequent category
     data['emp_title'].fillna(value=most_frequent, inplace=True)
     
     missing =  missing_values_table(data)
     print(missing)
     cols,num_cols,cat_cols =  row_cols_df(data)
 

     # Replace outlier with the means.
     # Compute the z-score of each value in the dataframe
     
     # Converting categorical data to numeric with factorize
     z = stats.zscore(data[num_cols])

     # Set the threshold for the maximum absolute z-score
     threshold = 3

     # Create a boolean mask that is True for values with an absolute z-score less than the threshold
     mask = np.abs(z) < threshold

     # Clean the dataset outliers with Z-score
     clean_data = data[num_cols][mask]
     where = clean_data.isnull().sum()>0
     where = num_cols[where]
     for col in where:
        mean = clean_data[col].mean()
        clean_data[col].fillna(mean, inplace=True)
     clean_data =  pd.concat((clean_data,data[cat_cols]),axis=1)

 
##################### MAKE 5 VISUALIZATIONS OF THE DATA #######################
     
     data_5_Visualization(clean_data)
     
     
############ SPLIT THE DATA AND USE mRMR FOR DIMENSION REDUCTION ##############
     clean_data = clean_data.drop(columns=['loan_id','issue_date'])
     cols,num_cols,cat_cols =  row_cols_df(clean_data)
     new_data=clean_data.copy()
     
     # Converting categorical data to numeric with factorize
     for col in cat_cols:
        new_data[col] = pd.factorize(clean_data[col])[0]
    
     X = new_data
     X = X.drop(columns=['interest_rate'])
     y = new_data['interest_rate']
     
  
     # Split the data for dimension reduction
     X_train, X_test, Y_train, Y_test = train_test_split( np.asarray(X), np.asarray(y), test_size=0.3, random_state=0, shuffle=True)
 
     # Dimension Reduction Technique mRMR for feature selection.
     features = mrmr_regression(X=pd.DataFrame(X_train), y=pd.DataFrame(Y_train), K=4)
     selected_features = cols[features] #[34, 13, 47, 20]
 
     
     # Scale the reduced data with Standard Scale
     # Predict the interest_rate with 2 Regression algorithms 
     final_data = new_data[selected_features]
     X = final_data
     y = new_data['interest_rate']
     
     
########### A SCATTER PLOT AND A HEATMAP FOR THE REDUCED VARIABLES ############

     # Plot a pair-Scatterplot and a Heatmap for the Reduced Variables
     explanatory_Analysis_Visualization( pd.concat((X,y),axis=1))
    
     
    
########################### Linear Simple  REGRESSOR ###########################
         
     standard_scaler = StandardScaler()
     X_train_scaled = standard_scaler.fit_transform(X_train)
     X_test_scaled = standard_scaler.fit_transform(X_test)

     regr = LinearRegression() 
     regr.fit(X_train_scaled, Y_train)
     y_pred = regr.predict(X_test_scaled)
     adjR2 , MSE  ,NRMSE,standar_residuals= performance_Metrics(np.asarray(Y_test), y_pred,np.asarray(X_test_scaled))
     linear_adjR2 = adjR2
     linear_y_pred = y_pred
     linear_y_resid = standar_residuals
     #results_Visualization(Y_test,y_pred,standar_residuals)


############################ RANDOM FOREST REGRESSOR ##########################


     # No, scaling is not necessary for Random forests,Gradient Boosting and Decision Trees
     # Tree-based model and hence does not require feature scaling.
      
     #Create the random forest model with 300 trees and a max depth of the trees of 6
     regr = RandomForestRegressor(n_estimators=100,max_depth=14)
     
     #fit the model with our training data
     regr.fit(X_train, Y_train)
     y_pred = regr.predict(X_test)
     adjR2 , MSE  ,NRMSE,standar_residuals= performance_Metrics(np.asarray(Y_test), y_pred,np.asarray(X_test))
     rfr_adjR2 = adjR2
     rfr_y_pred = y_pred
     rfr_y_resid = standar_residuals


######################## TEST RESULTS AND VISUALIZATIONS ######################
     results_Visualization(Y_test,rfr_y_pred,linear_y_pred,rfr_y_resid,linear_y_resid)

 
     print('\n The first regression algorithm is the Simple Linear Regression with adjR2: ',linear_adjR2)
     print('\n The second regression algorithm is the Random Forest Regression with adjR2: ',rfr_adjR2)
 
 