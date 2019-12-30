# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:38:17 2019

@author: Abhay
"""

import pandas as pd
import numpy as np

df=pd.read_csv('Credit_default_dataset.csv')
df.head(2)

df=df.drop(['ID'],axis=1)
df.head(2)

df.rename(columns={'PAY_0':'PAY_1'},inplace=True)

df['EDUCATION'].value_counts()

df['EDUCATION']= df['EDUCATION'].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})

df['MARRIAGE'].value_counts()

df['MARRIAGE']= df['MARRIAGE'].map({0:3,1:1,2:2,3:3})

from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
X=df.drop(['default.payment.next.month'],axis=1)
X=scaling.fit_transform(X)

y=df.iloc[:,-1]

# hyper parameter optimization

para={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }
    
 # hyper parameter optimization using randomizedsearchcv   
 
from xgboost import XGBClassifier
xgb=XGBClassifier()
from sklearn.model_selection import RandomizedSearchCV
rsc= RandomizedSearchCV(xgb,param_distributions=para,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
rsc.fit(X,y)
 
rsc.best_estimator_
 
rsc.best_params_
 
xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.4,
              learning_rate=0.3, max_delta_step=0, max_depth=3,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
 
from sklearn.model_selection import cross_val_score
cvs=cross_val_score(xgb,X,y,cv=10)
 
cvs.mean()