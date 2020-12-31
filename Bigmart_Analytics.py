# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

pd.set_option("max_columns",None)
pd.set_option("max_rows",None)

# Reading the data

Big_train = pd.read_csv("Bigmart Train.csv")
Big_test = pd.read_csv('Bigmart Test.csv')
Big_train.shape
Big_test.shape

# Understanding the data

Big_train.info()
Big_train.describe()
Big_train.columns

Big_train.nunique()
Big_test.nunique()

Big_train['Item_Outlet_Sales'].describe()
Big_train['Item_Outlet_Sales'].isnull().sum()
sns.distplot(Big_train['Item_Outlet_Sales'])

Big_train['Item_Weight'].describe()
Big_train['Item_Weight'].isnull().sum()
Big_train['Item_Weight'].fillna(Big_train['Item_Weight'].mean(),inplace=True)
sns.distplot(Big_train['Item_Weight'])
Big_test['Item_Weight'].describe()
Big_test['Item_Weight'].isnull().sum()
Big_test['Item_Weight'].fillna(Big_test['Item_Weight'].mean(),inplace=True)

Big_train['Outlet_Size'].describe()
Big_train['Outlet_Size'].unique()
Big_train['Outlet_Size'].value_counts()
Big_train['Outlet_Size'].fillna(value='Medium',inplace=True)
Big_train['Outlet_Size'].isnull().sum()
Big_train['Outlet_Size'].unique()
Big_train['Outlet_Size'] = Big_train['Outlet_Size'].replace([ 'Small','Medium','High'],[1,2,3])
sns.boxplot(Big_train['Outlet_Size'])
Big_test['Outlet_Size'].describe()
Big_test['Outlet_Size'].isnull().sum()
Big_test['Outlet_Size'].fillna(value='Medium',inplace=True)
Big_test['Outlet_Size'].unique()
Big_test['Outlet_Size'] = Big_test['Outlet_Size'].replace([ 'Small','Medium','High'],[1,2,3])

Big_train['Item_Visibility'].describe()
sns.distplot(Big_train['Item_Visibility'])
Big_train['Item_Visibility'].isnull().sum()
Big_test['Item_Visibility'].describe()
Big_test['Item_Visibility'].isnull().sum()

Big_train['Outlet_Establishment_Year'].describe()
Big_train['Outlet_Establishment_Year'].unique()
Big_train['Outlet_Establishment_Year'].value_counts()
Big_train['Outlet_Establishment_Year'] = 2009 - Big_train['Outlet_Establishment_Year']
Big_train['Outlet_Establishment_Year'].describe()
Big_test['Outlet_Establishment_Year'].describe()
Big_test['Outlet_Establishment_Year'].unique()
Big_test['Outlet_Establishment_Year'].value_counts()
Big_test['Outlet_Establishment_Year'] = 2009 - Big_test['Outlet_Establishment_Year']

Big_train['Item_Type'].value_counts()
Big_train['Item_Type'].isna().sum()
len(Big_train['Item_Type'].unique())
Big_test['Item_Type'].value_counts()
Big_test['Item_Type'].isna().sum()
len(Big_test['Item_Type'].unique())

Big_train['Item_Identifier'].describe()
len(Big_train['Item_Identifier'].unique())
Big_train['Item_Identifier'].value_counts()
Big_train['New_Item_Identifier'] = Big_train['Item_Identifier'].apply(lambda x: x[0:2])
Big_train['New_Item_Identifier'].value_counts()
Big_train['New_Item_Identifier'].head()
Big_train['New_Item_Identifier'].unique()
Big_train['New_Item_Identifier'] = Big_train['New_Item_Identifier'].replace(['FD','DR','NC'],[1,2,3])
Big_test['Item_Identifier'].describe()
len(Big_test['Item_Identifier'].unique())
Big_test['Item_Identifier'].value_counts()
Big_test['New_Item_Identifier'] = Big_test['Item_Identifier'].apply(lambda x: x[0:2])
Big_test['New_Item_Identifier'].value_counts()
Big_test['New_Item_Identifier'].head()
Big_test['New_Item_Identifier'].unique()
Big_test['New_Item_Identifier'] = Big_test['New_Item_Identifier'].replace(['FD','DR','NC'],[1,2,3])

Big_train['Item_Fat_Content'].value_counts()
Big_train['Item_Fat_Content'] = Big_train['Item_Fat_Content'].replace(['LF','low fat'],'Low Fat')
Big_train['Item_Fat_Content'] = Big_train['Item_Fat_Content'].replace('reg','Regular')
Big_train['Item_Fat_Content'].value_counts()
Big_train['Item_Fat_Content'].unique()
Big_train['Item_Fat_Content'] = Big_train['Item_Fat_Content'].replace(['Low Fat','Regular'],[1,2])
Big_test['Item_Fat_Content'].value_counts()
Big_test['Item_Fat_Content'] = Big_test['Item_Fat_Content'].replace(['LF','low fat'],'Low Fat')
Big_test['Item_Fat_Content'] = Big_test['Item_Fat_Content'].replace('reg','Regular')
Big_test['Item_Fat_Content'].value_counts()
Big_test['Item_Fat_Content'].unique()
Big_test['Item_Fat_Content'] = Big_test['Item_Fat_Content'].replace(['Low Fat','Regular'],[1,2])

pd.pivot_table(data=Big_train,values='Item_Outlet_Sales',
               index=['New_Item_Identifier','Item_Fat_Content'],columns=['Item_Type'])

Big_train['Item_MRP'].describe()
Big_train['Item_MRP'].isna().any()
sns.distplot(Big_train['Item_MRP'])
Big_test['Item_MRP'].describe()
Big_test['Item_MRP'].isna().any()

Big_train['Outlet_Identifier'].describe()
Big_train['Outlet_Identifier'].isna().any()
Big_train['New_Outlet_Identifier'] = Big_train['Outlet_Identifier'].apply(lambda x:x[4:]).astype(int)
Big_train['New_Outlet_Identifier'].head()
Big_test['Outlet_Identifier'].describe()
Big_test['Outlet_Identifier'].isna().any()
Big_test['New_Outlet_Identifier'] = Big_test['Outlet_Identifier'].apply(lambda x:x[4:]).astype(int)
Big_test['New_Outlet_Identifier'].head()

Big_train['Outlet_Location_Type'].describe()
Big_train['Outlet_Location_Type'].isna().any()
Big_train['Outlet_Location_Type'] = Big_train['Outlet_Location_Type'].apply(lambda x: x.split(' ')[1])
Big_train['Outlet_Location_Type'].head()
Big_test['Outlet_Location_Type'].describe()
Big_test['Outlet_Location_Type'].isna().any()
Big_test['Outlet_Location_Type'] = Big_test['Outlet_Location_Type'].apply(lambda x: x.split(' ')[1])
Big_test['Outlet_Location_Type'].head()

Big_train['Outlet_Type'].describe()
Big_train['Outlet_Type'].isna().any()
Big_train['Outlet_Type'] = Big_train['Outlet_Type'].apply(lambda x: x.split(' ')[1])
Big_train['Outlet_Type'].head()
Big_train['Outlet_Type'].unique()
Big_train['Outlet_Type'] = Big_train['Outlet_Type'].replace(['Type1','Type2','Type3','Store'],[1,2,3,4])
Big_test['Outlet_Type'].describe()
Big_test['Outlet_Type'].isna().any()
Big_test['Outlet_Type'] = Big_test['Outlet_Type'].apply(lambda x: x.split(' ')[1])
Big_test['Outlet_Type'].head()
Big_test['Outlet_Type'].unique()
Big_test['Outlet_Type'] = Big_test['Outlet_Type'].replace(['Type1','Type2','Type3','Store'],[1,2,3,4])

Big_train = pd.get_dummies(Big_train,columns=['Item_Type','Outlet_Location_Type'])
Big_test = pd.get_dummies(Big_test,columns=['Item_Type','Outlet_Location_Type'])
Big_test1 = Big_test.drop(['Item_Identifier','Outlet_Identifier'],axis=1)

# EDA

plt.figure(figsize=(15,10))
sns.countplot(Big_train['Item_Type'],data=Big_train)
sns.countplot(Big_train['Outlet_Size'],data=Big_train)
sns.countplot(Big_train['Outlet_Location_Type'],data=Big_train)
sns.countplot(Big_train['Outlet_Type'],data=Big_train)
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',data=Big_train)
sns.pairplot(vars=['Item_Identifier','Item_Visibility','Item_Type','Item_MRP','Item_Outlet_Sales'],data=Big_train)
sns.scatterplot(x=Big_train['Item_Weight'],y=Big_train['Item_Outlet_Sales'],hue='Item_Type',data=Big_train)
sns.scatterplot(x=Big_train['Outlet_Type'],y=Big_train['Item_Outlet_Sales'],hue='Item_Type',data=Big_train)
sns.boxplot(x=Big_train['Outlet_Type'],y=Big_train['Item_Outlet_Sales'],hue='Item_Type',data=Big_train)
sns.barplot(x= 'Item_Fat_Content',y='Item_Outlet_Sales',hue='Item_Type',data=Big_train)

Big_train.corr()['Item_Outlet_Sales'].sort_values(ascending=False)

# Train Test Split

X = Big_train.drop(['Outlet_Identifier','Item_Identifier','Item_Outlet_Sales'],axis=1)
y = Big_train['Item_Outlet_Sales']

SC = StandardScaler()
X = SC.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##########################################################
lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train,y_train)
y_prediction = lin_reg.predict(X_test)
y_prediction
lin_reg.score(X_train,y_train)
lin_reg.score(X_test,y_test)
###########################################################
reg_rf = RandomForestRegressor(max_depth=3)
reg_rf.fit(X_train,y_train)
Y_prediction1 = reg_rf.predict(X_test)
Y_prediction1
Y_prediction2 = reg_rf.predict(Big_test1)
Y_prediction2
Y_prediction2.shape
reg_rf.score(X_train,y_train)*100
reg_rf.score(X_test,y_test)*100
############################################################
lg = lgb.LGBMRegressor()
lg.fit(X_train,y_train)
Y_prediction10 = lg.predict(X_test)
Y_prediction10
Y_prediction20 = reg_rf.predict(Big_test1)
Y_prediction20
Y_prediction20.shape
lg.score(X_train,y_train)*100
lg.score(X_test,y_test)*100
##############################################################
Big_test['Item_Outlet_Sales'] = Y_prediction20
data = Big_test[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
data.head()
data.to_csv('lgb.csv')
###############################################################