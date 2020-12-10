import numpy as np
import pandas as pd
from datetime import datetime

#DATA CLEANING GENERIC METHODS
def missing_bool(df):
    #Return True if there are missing values
    return df.isna().values.any()

def frequency_missing(df, category):
    #Return absolute or relative (by columns) frequency of missing values
    if category in ['absolute', 'relative']:
        if category == 'absolute':
            return df.isna().sum().sum()
        else:
            return df.isna().sum()
    print('Category must be "absolute" or "relative"')
    return None

def percent_missing(df):
    temp = df.isna().sum()/(len(df))*100
    return temp

def missing(df):
    #Return the columns with more and less missing values
    temp = df.isna().sum()/(len(df))*100
    return temp.min(), temp.max()

def columns_rows_missing(df):
    #Print missing rows and columns
    #return df.dropna(), df.loc[:, df.isnull().any()].columns
    print(df.dropna())
    print(df.loc[:, df.isnull().any()].columns)
    

#DATA CLEANING PARTICULAR USECASE HOUSEHOLD PULSE SURVEY
def add_dates_start_end_inplace(data):
    data['DATE_START'] = data['WEEK'].apply(lambda idx: DATES_FROM_WEEK_IDX[idx][0])
    data['DATE_END']   = data['WEEK'].apply(lambda idx: DATES_FROM_WEEK_IDX[idx][1])

def add_age_inplace(data):
    data['AGE'] = datetime.now().year - data['TBIRTH_YEAR']

def replace_codes_with_nan_inplace(data):
    data.replace(-88, np.nan, inplace=True)
    data.replace(-99, np.nan, inplace=True)
    
def weekly_analysis(data):
    #Some surveyed people are in more than one week. 
    #If the variable analyzed didn't change, it counts 
    #as a duplicate
    data.drop_duplicates(inplace=True)
    #Incorporate dates for better reference in visualizations
    add_dates_start_end_inplace(data)
    
