import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns


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

def shopping_behaviors_correlation(df, location, name_location):
    corr = df[df.EST_MSA == location].loc[:, ['CHNGHOW1', 'CHNGHOW2', 'CHNGHOW3', 'CHNGHOW4', 
                                              'CHNGHOW5', 'CHNGHOW6', 'CHNGHOW7']].replace({np.nan: 0}).corr()
    xticklabels = ['ONLINE', 'PICK-UP', 'IN-STORE', 'CONTACTLESS PAYMENT', 'CASH', 
                   'AVOIDED DINING', 'RESUMED DINING']
    yticklabels = ['ONLINE', 'PICK-UP', 'IN-STORE', 'CONTACTLESS PAYMENT', 'CASH', 
                   'AVOIDED DINING', 'RESUMED DINING']
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(8.5, 8.5))
        ax = sns.heatmap(corr, 
                         mask=mask, 
                         vmax=.3, 
                         square=True, 
                         cmap="coolwarm_r", 
                         xticklabels=xticklabels, 
                         yticklabels=yticklabels, 
                         annot=True, 
                         fmt='1.2f',
                         annot_kws={"size": 15})
        
    _ = plt.title('Correlation Shopping Behavioral variables\n'+name_location, size=16, )

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

def cramers_matrix(df, location,name_location, shopping_variable, name_variable):
    corr = []
    demographics = ['WEEK', 'EGENDER', 'RHISPANIC', 
                    'RRACE', 'GENERATION',
                    'EEDUC', 'MS', 'THHLD_NUMPER', 
                    'THHLD_NUMKID', 'THHLD_NUMADLT']
    
    df = df[df.EST_MSA == location].loc[:, [shopping_variable] + demographics].replace({np.nan: 0})

    for var in demographics:
        c = cramers_v(df[shopping_variable], df[var])
        corr.append(c)
    a = pd.DataFrame(corr, index=demographics)
    a = a.rename(columns={0:shopping_variable})
    a['ABS_CHNGHOW'] = abs(a[shopping_variable]) 
    varsx = list(a['ABS_CHNGHOW'][1:].sort_values(ascending=False)[:3].index)

    print('Variables more correlated to '+name_variable+': \n{}'.format(a.loc[varsx, :][shopping_variable]))

    a.drop(columns=['ABS_CHNGHOW'], inplace=True)
    mask = np.zeros_like(a)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(a, mask=mask, square=True, cmap="coolwarm_r", annot=True, fmt='1.2f', annot_kws={"size": 15})
        _ = plt.title('Correlation Demographics and '+name_variable+'\n'+name_location, size=16)

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
            f, ax = plt.subplots(figsize=(7, 7))
        ax = sns.heatmap(temp, mask=mask, vmax=vmax, vmin=0, square=True, cmap="coolwarm_r", annot=True, fmt='1.2f', annot_kws={"size": 15})
        _ = plt.title('Correlation Demographics and '+name_variable+'\n'+name_location, size=16)