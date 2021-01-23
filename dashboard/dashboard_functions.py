import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os


def generation_(TBIRTH_YEAR):
# Function to determine generation by birthday year. 
# According to the survey, all persons born before 2002.
    if TBIRTH_YEAR in list(range(1946, 1955)):
        return 'Baby Boomer'
    elif TBIRTH_YEAR in list(range(1955, 1965)):
        return 'Generation Jones'
    elif TBIRTH_YEAR in list(range(1965, 1981)):
        return 'Generation X'
    elif TBIRTH_YEAR in list(range(1981, 1997)):
        return 'Millennials'
    elif TBIRTH_YEAR in list(range(1997, 2010)):
        return 'Generation Z'
    else:
        return 'Silent Generation'

def shopping_behaviors_correlation(df, location, location_name):
    df1 = df[df.EST_MSA == location].drop(columns=['EST_MSA'])
    corr = df1.replace({np.nan: 0}).corr()

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(8.5, 8.5))
        _ = plt.rcParams.update({'font.size': 22})
        ax = sns.heatmap(corr, 
                         mask=mask, 
                         vmax=.3, 
                         square=True, 
                         cmap="coolwarm_r", 
                         annot=True, 
                         fmt='1.2f',
                         annot_kws={"size": 22})
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 22)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 22)

    if not os.path.exists('images/'+location_name):
        os.makedirs('images/'+location_name)
    f.savefig('images/'+location_name+'/shopping_behaviors_correlation.png')

def cross_variable_correlation(df, location, name_location, shopping_variable, name_variable):
    l = ['SCRAM', 'WEEK', 'TBIRTH_YEAR', 'EGENDER', 'RHISPANIC', 'RRACE', 
         'EEDUC', 'MS', 'THHLD_NUMPER', 'THHLD_NUMKID', 'THHLD_NUMADLT']

    corr = df[df.EST_MSA == location].loc[:, [shopping_variable] + l].replace({np.nan: 0}).corr()
    corr['ABS_CHNGHOW'] = abs(corr[shopping_variable])
    varsx = list(corr['ABS_CHNGHOW'][1:].sort_values(ascending=False)[:3].index)

    print('Variables more correlated to '+name_variable+': \n{}'.format(corr.loc[varsx, :][shopping_variable]))
    
    #_ = sns.heatmap(corr, vmin=-0.2, vmax=0.2, cmap="YlGnBu")
    #_ = plt.show()
    corr.drop(columns=['ABS_CHNGHOW'], inplace=True)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap="coolwarm_r", annot=True, fmt='1.2f', annot_kws={"size": 15})
        _ = plt.title('Correlation Demographics and '+name_variable+'\n'+name_location, size=16)

def summary_shopping_behavior(df_shopping_statistics_msa, EST_MSA):
    temp = df_shopping_statistics_msa[df_shopping_statistics_msa.EST_MSA == EST_MSA]
    print('PURCHASE METHODS --------------------------------------------------------')
    print('Pct surveyed people doing more online-purchases: {}\n'.format(temp.Online))
    print('Pct surveyed people doing more pickup-purchases: {}\n'.format(temp.Pickup))
    print('Pct surveyed people doing more online + pickup purchases: {}\n'.format(temp['Online + Pickup']))
    print('Pct surveyed people doing more in-store purchases: {}\n'.format(temp['In-store']))
    
    print('PAYMENT METHODS ---------------------------------------------------------')
    print('Pct surveyed people using contactless payment methods: {}\n'.format(temp['CHNGHOW4']))
    print('Pct surveyed people using more cash: {}\n'.format(temp['CHNGHOW5']))
    
    print('RESTAURANTS -------------------------------------------------------------')
    print('Pct surveyed people who resumed eating at restaurants: {}\n'.format(temp['CHNGHOW7']))
    print('Pct surveyed people who avoided eating at restaurants: {}\n'.format(temp['CHNGHOW6']))

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def cramers_matrix(df, location, location_name, shopping_variable, name_variable):
    corr = []
    demographics = ['EGENDER', 'RHISPANIC', 
                    'RRACE', 'GENERATION',
                    'EEDUC', 'MS', 'THHLD_NUMPER', 
                    'THHLD_NUMKID', 'THHLD_NUMADLT']
    
    df = df[df.EST_MSA == location].loc[:, [shopping_variable] + demographics].replace({np.nan: 0})
    df.rename({shopping_variable: name_variable}, axis=1, inplace=True)

    for var in demographics:
        c = cramers_v(df[name_variable], df[var])
        corr.append(c)
    a = pd.DataFrame(corr, index=demographics)
    a = a.rename(columns={0:name_variable})
    a['ABS_CHNGHOW'] = abs(a[name_variable]) 
    varsx = list(a['ABS_CHNGHOW'][1:].sort_values(ascending=False)[:3].index)

    print('Variables more correlated to '+name_variable+': \n{}'.format(a.loc[varsx, :][name_variable]))

    a.drop(columns=['ABS_CHNGHOW'], inplace=True)
    mask = np.zeros_like(a)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        _ = plt.rcParams.update({'font.size': 22})
        ax = sns.heatmap(a, square=True, cmap="coolwarm_r", annot=True, fmt='1.2f', annot_kws={"size": 22})
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 22)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 22)
       
    if not os.path.exists('images/'+location_name):
        os.makedirs('images/'+location_name)
    f.savefig('images/'+location_name+'/demographics_'+name_variable+'_correlation.png')

def demographics_shopping_correlation(df, location, name_location, shopping_variable, name_variable, sub_list_demographics):
    
    temp = df[df.EST_MSA == location].loc[:, [shopping_variable] + sub_list_demographics].replace({np.nan: 0})

    temp = pd.get_dummies(data=temp, columns=sub_list_demographics).corr()
    vmax = temp.loc[:, shopping_variable].sort_values(ascending=False)[1]
    mask = np.zeros_like(temp)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        if len(sub_list_demographics) > 2:
            f, ax = plt.subplots(figsize=(20, 20))
        elif len(sub_list_demographics) == 2:
            f, ax = plt.subplots(figsize=(15, 15))
        else:
            f, ax = plt.subplots(figsize=(7.5, 7.5))
        ax = sns.heatmap(temp, mask=mask, vmax=vmax, vmin=0, square=True, cmap="coolwarm_r", annot=True, fmt='1.2f', annot_kws={"size": 15})
        _ = plt.title('Correlation Demographics and '+name_variable+'\n'+name_location, size=16)

def indexed_mobility_average_person(df, location):
    df_index = df[df.NAME == location].loc[:, '2020-03-01':]

    df_index = pd.melt(df_index, var_name='date', value_name='index')
    df_index['date'] = pd.to_datetime(df_index['date'])
    df_index['city'] = location
    
    f, ax = plt.subplots(figsize=(15.5, 4.5))
    _ = plt.rcParams.update({'font.size': 16})
    _ = sns.lineplot(data=df_index, x='date', y='index')
    _ = plt.axhline(y=100, color='k', linestyle='--', label='Baseline', lw=0.5)
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Indexed mobility')
    _ = plt.title('Indexed mobility for a regular person \n'+location)   

def mobility_average_person(df, location):
    df_kms= df[df.NAME == location].loc[:, '2020-03-01':]

    df_kms = pd.melt(df_kms, var_name='date', value_name='kms')
    df_kms['date'] = pd.to_datetime(df_kms['date'])
    df_kms['city'] = location
    
    f, ax = plt.subplots(figsize=(15.5, 4.5))
    _ = plt.rcParams.update({'font.size': 16})
    _ = sns.lineplot(data=df_kms, x='date', y='kms')
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Kms')
    _ = plt.title('Mobility for a regular person \n'+location)   

def mobility_venues_foursquare(df, location):
    df = df[[location,'class']]
    df.index = pd.to_datetime(df.index)
    
    temp = df.groupby('class').count().reset_index()
    max_data = max(temp[location])
    #only display historical data with 80% of the longer serie
    valid_venues = list(temp[temp[location] >= 0.8*max_data]['class'])
    
    f, ax = plt.subplots(figsize=(15.5, 4.5))

    for venue in valid_venues:
        temp = df[df['class'] == venue].loc[:, location]
        _ = sns.lineplot(data=temp, label=venue)
    
    _ = ax.legend(bbox_to_anchor=(1.05, 1.05))
    _ = plt.rcParams.update({'font.size': 16})
    _ = plt.axhline(y=100, color='k', linestyle='--', label='Baseline', lw=0.5)
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Indexed Mobility')
    _ = plt.title('Mobility trends to different venues in \n'+location)   

def mobility_venues_google(df, location):
    df = df[df.NAME == location]
    df1 = pd.melt(df, id_vars=['NAME', 'category'], var_name='Date')
    df1.Date = pd.to_datetime(df1.Date)
    temp = df1.groupby('category').count().reset_index()

    max_data = max(temp['value'])
    #only display historical data with 80% of the longer serie
    valid_venues = list(temp[temp['value'] >= 0.8*max_data]['category'])

    f, ax = plt.subplots(figsize=(15.5, 4.5))

    for venue in valid_venues:
        temp = df1[df1['category'] == venue]
        _ = sns.lineplot(data=temp, label=venue, y='value', x='Date')

    _ = ax.legend(bbox_to_anchor=(1.05, 1.05))
    _ = plt.rcParams.update({'font.size': 16})
    _ = plt.axhline(y=0, color='k', linestyle='--', label='Baseline', lw=0.5)
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Indexed Mobility')
    _ = plt.title('Mobility trends to different venues in \n'+location)   

def apple_mobility_cities(df, location):
    df_apple_sf = df[df.city == location]
    df_apple_sf.date = pd.to_datetime(df_apple_sf.date)

    f, ax = plt.subplots(figsize=(14.5, 4))
    _ = sns.lineplot(data=df_apple_sf, label='Driving', y='drv', x='date')
    _ = sns.lineplot(data=df_apple_sf, label='Transit', y='trn', x='date')
    _ = sns.lineplot(data=df_apple_sf, label='Walking', y='wlk', x='date')

    _ = ax.legend(bbox_to_anchor=(1.05, 1.05))
    _ = plt.rcParams.update({'font.size': 16})
    _ = plt.axhline(y=100, color='k', linestyle='--', label='Baseline', lw=0.5)
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Indexed Mobility')
    _ = plt.title('Mobility trends in \n'+location)   

def serie_decomposition(df, colname, model, display, location):

    df = df[df.city == location]
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)

    if df[colname].isnull().any():
        df.fillna(method='ffill', inplace=True)
    result_add = seasonal_decompose(df[colname], model=model, extrapolate_trend='freq')
    df_reconstructed = pd.concat([result_add.seasonal, result_add.trend, result_add.resid, result_add.observed], axis=1)
    df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
    
    
    if display:
        plt.rcParams.update({'figure.figsize': (12,10)})
        result_add.plot().suptitle('Additive Decompose', fontsize=16)
        plt.show()
        
    return df_reconstructed

def compile_decomposition(df, variable_decomposition, location):
# Display a plot with the selected variable decomposition for driving, walking and transit
    out1 = serie_decomposition(df, 'trn', 'additive', False, location)

    out2 = serie_decomposition(df, 'drv', 'additive', False, location)

    out3 = serie_decomposition(df, 'wlk', 'additive', False, location)

    _ = plt.figure(figsize=(15.5, 4.5))

    if variable_decomposition == 'trend':
        _ = plt.plot(out1.trend, label='Transit routes')
        _ = plt.plot(out2.trend, label='Driving routes')
        _ = plt.plot(out3.trend, label='Walking routes')
    elif variable_decomposition == 'residual':
        _ = plt.plot(out1.resid, label='Transit routes')
        _ = plt.plot(out2.resid, label='Driving routes')
        _ = plt.plot(out3.resid, label='Walking routes')
    else:
        _ = plt.plot(out1.seas, label='Transit routes')
        _ = plt.plot(out2.seas, label='Driving routes')
        _ = plt.plot(out3.seas, label='Walking routes')

    _ = plt.axhline(y=100, color='k', linestyle='--', label='Baseline, January 13th, 2020', lw=0.5)
    _ = plt.title('Route requests '+variable_decomposition+' after timeserie decomposition \nby type of transportation in '+location, size=16)
    _ = plt.xlabel('Date', size=12)
    _ = plt.ylabel('Route requests', size=12)
    _ = plt.legend(('Transit routes', 'Driving routes', 'Walking routes'))

def transaction_cleaning(transaction):

    l1 = ['restaurant_reservation', 'pickup', 'delivery']
    l2 = ['restaurant_reservation', 'pickup']
    l3 = ['restaurant_reservation', 'delivery']
    l4 = ['pickup']
    l5 = ['pickup', 'delivery']
    l6 = ['delivery']
    l7 = ['restaurant_reservation']

    try:
        t = transaction[1:-1].replace("'", "").replace(" ", "").split(',')
        set1 = set(t)

        if set(l1).intersection(set1) == set(l1) and set(l1).intersection(set1) == set1:
            return l1
        elif set(l2).intersection(set1) == set(l2) and set(l2).intersection(set1) == set1:
            return l2
        elif set(l3).intersection(set1) == set(l3) and set(l3).intersection(set1) == set1:
            return l3
        elif set(l4).intersection(set1) == set(l4) and set(l4).intersection(set1) == set1:
            return l4
        elif set(l5).intersection(set1) == set(l5) and set(l5).intersection(set1) == set1:
            return l5
        elif set(l6).intersection(set1) == set(l6) and set(l6).intersection(set1) == set1:
            return l6
        elif set(l7).intersection(set1) == set(l7)and set(l7).intersection(set1) == set1:
            return l7
        else:
            return np.nan
    except:
        return np.nan

def convert2dollar(price):
#Return the maximum price of every bin
    if price == '$':
        return 10
    elif price == '$$':
        return 30
    elif price == '$$$':
        return 60
    elif price == '$$$$':
        return 120
    else:
        return np.nan

def check_pickup_delivery(purchase):
#Check if the clean purchase list contains pickup and delivery
    if purchase is np.nan:
        return np.nan
    
    if 'pickup' in purchase and 'delivery' in purchase:
        return 'pickup and delivery'
    elif 'pickup' in purchase and 'delivery' not in purchase:
        return 'only pickup'
    elif 'pickup' not in purchase and 'delivery' in purchase:
        return 'only delivery'
    else:
        return 'neither pickup nor delivery'