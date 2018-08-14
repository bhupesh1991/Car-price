# -*- coding: utf-8 -*-
"""


@author: Bhupesh
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.getcwd()

os.chdir("C:\\Users\\Bhupesh\\Downloads")

data=pd.read_csv("C:\\Users\\Bhupesh\\Downloads\\New Folder\\CarPrice_Assignment.csv")
data1=pd.read_csv("C:\\Users\\Bhupesh\\Downloads\\New Folder\\carname.csv")
df=data

df.describe()
df.info()
df.fueltype.value_counts()
df.fueltype=pd.get_dummies(df.fueltype)

df.aspiration.value_counts()

df.aspiration=pd.get_dummies(df.aspiration)

df.doornumber = pd.get_dummies(df.doornumber)

df.carbody.value_counts()

carbody=pd.DataFrame(df.carbody)
carbody=pd.get_dummies(carbody)
carbody=carbody.drop("carbody_convertible",axis=1)

df=pd.merge(df,carbody,how='left',right_index=True,left_index=True)

df=df.drop("carbody",axis=1)

df.drivewheel.value_counts()

drivewheel=pd.DataFrame(df.drivewheel)

drivewheel=pd.get_dummies(drivewheel)

drivewheel=drivewheel.drop("drivewheel_4wd",axis=1)

df=pd.merge(df,drivewheel,how='left',right_index=True,left_index=True)

df=df.drop("drivewheel",axis=1)

df.enginelocation.value_counts()

df.enginelocation=pd.get_dummies(df.enginelocation)


df.enginetype.value_counts()

enginetype=pd.DataFrame(df.enginetype)

enginetype=pd.get_dummies(enginetype)


enginetype=enginetype.drop("enginetype_dohc",axis=1)

df=pd.merge(df,enginetype,how='left',left_index=True,right_index=True)

df=df.drop("enginetype",axis=1)

df.cylindernumber.value_counts()

cylindernumber=pd.DataFrame(df.cylindernumber)

cylindernumber=pd.get_dummies(cylindernumber)

cylindernumber=cylindernumber.drop("cylindernumber_eight",axis=1)

df=pd.merge(df,cylindernumber,how='left',left_index=True,right_index=True)

df=df.drop("cylindernumber",axis=1)

df.fuelsystem.value_counts()

fuelsystem=pd.DataFrame(df.fuelsystem)

fuelsystem=pd.get_dummies(fuelsystem)

fuelsystem=fuelsystem.drop("fuelsystem_1bbl",axis=1)

df=pd.merge(df,fuelsystem,how='left',left_index=True,right_index=True)


df=df.drop("fuelsystem",axis=1)

#merging carname dummies variable

data1=data1.drop("Unnamed: 0",axis=1)

df=pd.merge(df,data1,how='left',right_index=True,left_index=True)

df=df.drop("car_ID",axis=1)

df=df.drop("CarName",axis=1)




#splitting the data into training and testing set after assining x  and y 

x=df.drop("price",axis=1)
y=pd.DataFrame(df.price)



vif=pd.DataFrame()
vif['vif factor']=[variance_inflation_factor(x.values,i)for i in range (x.shape[1])]
vif['feature']=x.columns
vif.round(1)
m=smf.OLS(y,x).fit()
m.summary()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

lm=linear_model.LinearRegression()
model=lm.fit(x_train,y_train)
prediction=lm.predict(x_test)

model.score(x_test,y_test)
