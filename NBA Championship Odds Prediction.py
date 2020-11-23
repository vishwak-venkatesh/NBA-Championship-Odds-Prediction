# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 00:11:50 2020

@author: vvenkatesh
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vvenkatesh\Desktop\Vishwak\NBA Analysis\past_rosters.csv")

#Drop ROTY
df=df.drop(['ROTY'],axis=1)

#Unbalanced classes:
# 0 is not champion, 1 is champion
print(df.Champion.value_counts())

labels = np.array(df.pop('Champion'))
train, test, train_labels, test_labels = train_test_split(df,labels,stratify = labels, test_size= 0.25,random_state = 40) 

# Store train and test team details to reconstruct at the end
train_team=train.pop('Team')
train_year=train.pop('Year')
test_team=test.pop('Team')
test_year=test.pop('Year')

# Balanced Random Forest performs sampling to account for unbalanced classes
model = BalancedRandomForestClassifier( n_estimators=100, random_state=40,                  max_features = 'sqrt',n_jobs=-1, verbose = 1)

model.fit(train, train_labels)

#Show model importances
importances = model.feature_importances_*100
indices = np.argsort(importances)[::-1]
names = [train.columns[i] for i in indices]

# Barplot: Add bars
plt.bar(range(train.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(train.shape[1]), names, rotation=20, fontsize = 8)
plt.yticks(range(0,35,5), fontsize=12)
plt.grid(b=None,axis='x')
# Create plot title
plt.title("Feature Importances")
# Show plot
plt.show()

#Training data prediction
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

#Combine predicted train data odds with team name and year
train_1=pd.concat([train_year,train_team,train],axis=1)
train_1.reset_index(drop=True,inplace=True)
train_1=pd.concat([train_1,pd.DataFrame(train_labels)],axis=1)
train_1=train_1.rename(columns={0:'Champion'})
train_1=pd.concat([train_1,pd.DataFrame(train_rf_probs)],axis=1)
train_1=train_1.rename(columns={'Year':'Year','Team':'Team',0:'Probs'})

#train_1.sort_values(by=['Probs'],ascending=False)           

#Combine predicted test data odds with team name and year
test_1=pd.concat([test_year,test_team,test],axis=1)
test_1.reset_index(drop=True,inplace=True)
test_1=pd.concat([test_1,pd.DataFrame(test_labels)],axis=1)
test_1=test_1.rename(columns={0:'Champion'})
test_1=pd.concat([test_1,pd.DataFrame(rf_probs)],axis=1)
test_1=test_1.rename(columns={'Year':'Year','Team':'Team',0:'Probs'})

#test_1.sort_values(by=['Probs'],ascending=False)           



# Confusion matrix for test 
confusion_matrix(test_1['Champion'],rf_predictions)

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

#Evaluate model
evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)   

df_1=pd.concat([train_1,test_1],axis=0)
df_1.to_csv(r"C:\Users\vvenkatesh\Desktop\Vishwak\NBA Analysis\all_rosters_bal_rf_probs.csv",encoding="utf8")

# Visualizing distribution of predicted probabilities
plt.hist(df_1.Probs,bins=20)

# Get roster data for 2020 season
curr=pd.read_csv(r"C:\Users\vvenkatesh\Desktop\Vishwak\NBA Analysis\2020 roster.csv")

curr_team=curr.pop('Team')
curr_year=curr.pop('Year')
curr=curr.drop(['ROTY'],axis=1)

curr_rf_predictions = model.predict(curr)
curr_rf_probs = model.predict_proba(curr)[:, 1]

curr_1=pd.concat([curr_year,curr_team,curr],axis=1)
curr_1.reset_index(drop=True,inplace=True)
curr_1=pd.concat([curr_1,pd.DataFrame(curr_rf_probs)],axis=1)
curr_1=curr_1.rename(columns={0:'Probs'})

curr_1.sort_values(by=['Probs'],ascending=False)      

curr_1.to_csv(r"C:\Users\vvenkatesh\Desktop\Vishwak\NBA Analysis\curr_rosters_bal_rf_probs.csv",encoding="utf8")
     


