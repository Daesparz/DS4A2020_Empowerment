import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
                         fmt='1.3f')
        
    _ = plt.title('Correlation Shopping Behavior variables\n'+name_location, size=16)



def cross_variable_correlation(df, location, shopping_variable, name_variable, name_location):
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
        ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap="coolwarm_r", annot=True, fmt='1.2f')
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