#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 21:10:10 2021

@author: kavindrasahabir
"""
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing


df = pd.read_csv('~/Downloads/speeddating.csv', na_values = '?')

label = preprocessing.LabelEncoder()

'''

'''
df.dropna(inplace=True)


#for i in df:
    #normalize_series(df[i])

'''
df['attractive'] = normalize_series(df['attractive'])
df['samerace'] = normalize_series(df['samerace'])
df['age'] = normalize_series(df['age'])
df['age_o'] = normalize_series(df['ago_o'])
'''
#df['importance_same_race'] = df['importance_same_race'] /df['importance_same_race'].abs().max()



correlations = df.corr().unstack().sort_values(ascending=False).drop_duplicates()
#print(correlations)#

#create correlation heatmaps based on gender

df1 = df[['gender','age','age_o','d_age','samerace','match',
         'shared_interests_o','shared_interests_partner','funny_o','funny_partner',
         'attractive_o','attractive_partner','intelligence_partner',
         'intelligence_o','sincere_partner','sinsere_o','ambition','ambitous_o',
         'interests_correlate']]

df1.dropna(inplace=True)



#create a list of attributes that show what the person thinks of the other person


att = ['shared_interests_o','funny_o','attractive_o','intelligence_o','sinsere_o','ambitous_o']

#heatmap for men
#np.triu is to only have the relevant data

df1_male = df1[df1['gender']=='male']
#df_ma = sns.heatmap(df1_male[att].corr(), mask = np.triu(df1_male[att].corr()),linewidths = 0.1, linecolor='white')


#heatmap for women
df1_female = df1[df1['gender']=='female']
df_fem = sns.heatmap(df1_female[att].corr(), mask = np.triu(df1_female[att].corr()),linewidths = 0.1, linecolor='white')
df1['gender']= label.fit_transform(df['gender'])
#label value of 0 is female, value of 1 is male

def normalize_series(s):
    return (s - s.min()) / (s.max() - s.min())

df_norm = df1.apply(lambda x: normalize_series(x))




#using Naive Bayes classifier to predict matches

df_train = df_norm.sample(frac=2/5)
df_test = df_norm[~(df_norm.index.isin(df_train.index))]

df_train.head()

print(len(df_train), len(df_test), len(df1))

#function to find the number of instances for attribute per class

def find_counts(df, splitby):
    count = {}

    # determine values of dataset split per class
    value_counts = df[splitby].value_counts().sort_index()
    #convert to numpy array for elementwise calculations
    count["class_labels"] = value_counts.index.to_numpy()
    count["class_counts"] = value_counts.values

    # determine conditional probabilities for each attribute per class
    
    # for each element in each of the columns, excluding the Type column
    for i in df.drop(splitby, axis=1).columns:
        
        #create empty dictionaries to store the condiitonal probabilities for each attribute
        count[i] = {}
        

        # find number of values for each attribute for each class 
        counts = df.groupby(splitby)[i].value_counts()
        #print(counts)
        
        #pivot dictionary so that the dictionary is split into groups based on columns, 
        #and each column is each class 
        training_counts = counts.unstack(splitby)
        
        

        # table contains NaN values after pivoting 
        #(since not all values for each attribute exist in all the class labels)
        #therefore, we add a count of 1 to every value to solve this
        #this does not affect the final result as the ratios stay intact
        
        if training_counts.isna().any(axis=None):
           training_counts.fillna(value=0, inplace=True)
           training_counts += 1
        
        
        # calculate probabilities for each attribute belonging to each class
        df_probabilities = training_counts / training_counts.sum()
        
        #print (df_probabilities)
        #storing the probabilities in the empty dictionaries created with count[i]={} 
        for j in df_probabilities.index:
            probabilities = df_probabilities.loc[j]
            count[i][j] = probabilities
            #print(probabilities)
        
    return count
    
find_counts = find_counts(df_norm,"match")

#function for prediction
def predict_example(row, dict):
    
    #assigning class counts arrays to variable class_estimates
    class_estimates = dict["class_counts"]
    
    #for each index in dictionaries created to store values in 'find_counts' function
    for i in row.index:
        
        # need to handle cases where values are only found in the test dataset
        # so skip those cases and don't take them into consideration 
        #for final calculations
        try:
            value = row[i]
            probabilities = dict[i][value]
            class_estimates = class_estimates * probabilities

        
        except KeyError:
            continue
    
    #find the probability of each row's class label 
    #by finding the maximum probability calculated 
    #out of the three probabilities found for each row
    max_value = class_estimates.argmax()
    
    #calling the predicted class labels 
    prediction = dict["class_labels"][max_value]
    
    return prediction

# applying the function to the test dataframe 
#axis = 1 ensures that the function is applied to each row 
# args takes in find_counts for the row variable
predictions = df_test.apply(predict_example, axis=1, args=(find_counts,))

#check accuracy by appending predictions to testing dataframe 
#and checking how many rows are predicted correctly
df_test['predicted class label'] = predictions

accuracy = []
comparison_column = np.where(df_test["match"] == df_test["predicted class label"], True, False)
for i in comparison_column:
    if i == True:
        accuracy.append(1)









