# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:04:29 2020

@author: steff
"""

def dataframe_exploration(df, df_name_string):
    
    print("\n\n******************** BASIC EXPLORATION FOR DATAFRAME: '",df_name_string,"'")
    print('\nDATA TYPE:\n\n', type(df))
    print('\nDATA SHAPE:\n\n', df.shape)
    print('\nDATA INFO:\n')
    df.info()
    print('\nDUPLICATE DATA:\n\n', df.duplicated().sum())
    print('\nDATA MISSING:\n\n', df.isna().sum())
    print('\nUNIQUE VALUES:\n\n', df.nunique().sort_values())
    print('\nDATA HEAD:\n\n', df.head())

def target_variable_exploration(target, task):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if task == 'regression':
        plt.figure(figsize=(14,6))
        plt.subplot(1, 2, 1)
        sns.boxplot(target)
        plt.subplot(1, 2, 2)
        sns.distplot(target)
        plt.show()
    elif task == 'classification':
        sns.countplot(target)
    else:
        print('Invalid task parameter: Specify task = ''regression'' or ''classification''')
    
    print(target.describe())
    
def plot_feature(df, col, target):
# NOTE: features must be either of type 'object' or 'int64' to use this function
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16,5))
    plt.subplot(1, 2, 1)
    
    if df[col].nunique() > 12:
        sns.distplot(df[col])
        plt.ylabel('Distribution')
    
    else:
        sns.countplot(x=col, data=df)
        plt.ylabel('Counts')

    if df[col].dtype == 'object':
        plt.xticks(rotation=45)
    plt.xlabel('Feature Variable: ' + col) 
    plt.title('Countplot')
    
    plt.subplot(1, 2, 2)
    
    '''if df[col].dtype == 'object':
        df_copy = df.copy()
        df_copy[col] = df_copy[col].astype('category')
        order = df_copy.groupby(col)[target].mean().sort_values().index.tolist()
        df_copy[col].cat.reorder_categories(order, inplace=True)
        sns.boxplot(x=col, y=target, data=df_copy)
        plt.xticks(rotation=45)
    
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':'''
    counts = df.groupby(col)[target].count()
    means = df.groupby(col)[target].mean()
    sns.scatterplot(x=means.index, y=means.values, size=counts.values)

    plt.xlabel('Feature Variable: ' + col) 
    plt.ylabel('Target Variable: ' + target) 
    plt.title('Average target variable value for each feature value')           
    plt.show()
    
def dataframe_correlation_heatmap(df, target):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
# NOTE: catgorcial columns should be numerically encoded first, before any correlation can be infered     
# this function encodes the category with: mean of the category in the target variable
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            cats = df[col].unique()
            cat_dict = {}
            for cat in cats:
                cat_dict[cat] = df_copy[df_copy[col]==cat][target].mean()
            df_copy[col] = df_copy[col].map(cat_dict)
    c = df_copy.corr()
    mask = np.triu(np.ones_like(c, dtype=bool))
    
    plt.figure(figsize=(12,10))
    sns.heatmap(c, cmap='coolwarm', annot=True, mask=mask)
 
def object_to_dummy_variables(df, index, col, drop_og=False):
    
    import pandas as pd
    
    dummies = pd.get_dummies(df[col], prefix=col)
    df = df.merge(dummies, on=index)
    
    if drop_og == True:
        df.drop(columns=col, inplace=True)
    
    return df

def bin_continuous_feature(df, col, bins, labels):
    
    import pandas as pd
    
    new_col = col + '_binned'
    df[new_col] = pd.cut(df[col], bins, labels=labels, right=False)
    df[new_col] = df[new_col].astype('object')
    
    return df

def group_stats(df, groups, target):
    
    import pandas as pd
    
    # new group statistics
    df_grouped = df.groupby(groups)
    
    group_stats = pd.DataFrame({'group_count':df_grouped[target].count()})
    group_stats['group_mean'] = df_grouped[target].mean()
    group_stats['group_max'] = df_grouped[target].max()
    group_stats['group_min'] = df_grouped[target].min()
    group_stats['group_std'] = df_grouped[target].std()
    group_stats['group_median'] = df_grouped[target].median()
    group_stats.reset_index(inplace=True)

    return group_stats
    
def plot_classification_report(cr, n_classes, title='Classification report ', with_avg_total=False):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    lines = cr.split('\n')
    classes = []
    plotMat = []
    
    for line in lines[2 : n_classes+2]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')    
    
    