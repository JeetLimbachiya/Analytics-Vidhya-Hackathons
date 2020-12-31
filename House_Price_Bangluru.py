# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Lasso,LinearRegression,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,explained_variance_score

pd.set_option('max_columns',None)
pd.set_option('max_rows',None)

house = pd.read_csv('House_Price_Train.csv')
house_test = pd.read_csv('House_Price_Test.csv')
house.head()
house_test.shape
house.shape

house.info()
house.describe()

# Data clearing

house['area_type'].describe()
house['area_type'].value_counts()
house['area_type'].isnull().sum()
house['New_area_type'] = house['area_type'].copy()
house['New_area_type'].replace(['Super built-up  Area','Built-up  Area','Plot  Area','Carpet  Area'],[0,1,2,3],inplace=True)
house['New_area_type'].head()
sns.countplot(x='area_type',data=house)

house_test['area_type'].describe()
house_test['area_type'].value_counts()
house_test['area_type'].isnull().sum()
house_test['New_area_type'] = house_test['area_type'].copy()
house_test['New_area_type'].replace(['Super built-up  Area','Built-up  Area','Plot  Area','Carpet  Area'],[0,1,2,3],inplace=True)
house_test['New_area_type'].head()

house['size'].describe()
len(house['size'].unique()) #32
house['size'].unique()
house['size'].isnull().sum()
house['size'].fillna(method='ffill',inplace=True)
house['BHK'] = house['size'].apply(lambda x: int(x.split(' ')[0]))
house['BHK'].astype(float)

house_test['size'].describe()
house_test['size'].unique()
house_test['size'].isnull().sum()
house_test['size'].fillna(method='ffill',inplace=True)
house_test['BHK'] = house_test['size'].apply(lambda x: int(x.split(' ')[0]))
house_test['BHK'].astype(float)

house['location'].describe()
len(house['location'].unique()) # 1306
house['location'].unique()
house['location'].value_counts().sum()
house['location'].isnull().sum()
house['location'].fillna(method='ffill',inplace=True)
sns.countplot(house['location'],data=house)
location_values = house.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_values
New_location = location_values[location_values < 100]
New_location
house['location'] = house['location'].apply(lambda x: 'not_needed' if x in New_location else x)
len(house['location'].unique())
New_house_location = pd.get_dummies(house['location'])
house.shape
New_house_location.shape
house = pd.concat([house,New_house_location.drop('not_needed',axis = 1)],axis=1)
house.shape

house_test['location'].describe()
len(house_test['location'].unique())  #495
house_test['location'].value_counts().sum()
house_test['location'].isnull().sum()
location_values1 = house_test.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_values1
New_location41 = location_values1[location_values1 <13]
New_location41
house_test['location'] = house_test['location'].apply(lambda x: 'not_needed' if x in New_location41 else x)
len(house_test['location'].unique())
New_house_location41 = pd.get_dummies(house_test['location'])
house_test.shape
New_house_location41.shape
house_test = pd.concat([house_test,New_house_location41.drop('not_needed',axis = 1)],axis=1)
house_test.shape

house['bath'].describe()
house['bath'].unique()
house['bath'].isnull().sum()
house['bath'].fillna(method='ffill',inplace=True)
house['bath'].head()

house_test['bath'].describe()
house_test['bath'].unique()
house_test['bath'].isnull().sum()
house_test['bath'].fillna(method='ffill',inplace=True)
house_test['bath'].head()

house['total_sqft'].describe()
len(house['total_sqft'].unique())
house['total_sqft'].isnull().sum()
house['total_sqft'].head()
house['total_sqft'].tail()

def ts_to_nts(x):
    T = x.split('-')
    if len(T) == 2:
        return (float(T[0]) + float(T[1]))
    try:
        return float(x)
    except:
        return np.nan

ts_to_nts('30Acres')
ts_to_nts('2100-2850')
house['total_sqft'] = house['total_sqft'].apply(ts_to_nts)
house.iloc[1086]
house['total_sqft'].fillna(method='ffill',inplace=True)
house.iloc[1086]
house['total_sqft'].dtype
sns.distplot(house['total_sqft'])
house['total_sqft'] = house['total_sqft'].apply(lambda x: ((x-house['total_sqft'].min())/house['total_sqft'].max()+house['total_sqft'].min()))

house_test['total_sqft'].describe()
len(house_test['total_sqft'].unique())
house_test['total_sqft'].isnull().sum()
house_test['total_sqft'].head()
house_test['total_sqft'].tail()

def ts_to_nts(x):
    T = x.split('-')
    if len(T) == 2:
        return (float(T[0]) + float(T[1]))
    try:
        return float(x)
    except:
        return np.nan

ts_to_nts('34.46Sq. Meter')
ts_to_nts('2100-2850')
house_test['total_sqft'] = house_test['total_sqft'].apply(ts_to_nts)
house_test.iloc[42]
house_test['total_sqft'].fillna(method='ffill',inplace=True)
house_test.iloc[42]
house_test['total_sqft'].dtype
house_test['total_sqft'] = house_test['total_sqft'].apply(lambda x: ((x-house_test['total_sqft'].min())/house_test['total_sqft'].max()+house_test['total_sqft'].min()))

house['society'].describe()
len(house['society'].unique())   # 2688
house['society'].isnull().sum()
house['society'].fillna(method='ffill',inplace=True)

house_test['society'].describe()
len(house_test['society'].unique())   # 595
house_test['society'].isnull().sum()
house_test['society'].fillna(method='ffill',inplace=True)

house['balcony'].describe()
house['balcony'].isnull().sum()
house['balcony'].fillna(method='ffill',inplace=True)
house['balcony'].head()
sns.distplot(house['balcony'])

house_test['balcony'].describe()
house_test['balcony'].isnull().sum()
house_test['balcony'].fillna(method='ffill',inplace=True)
house_test['balcony'].head()
house_test.columns

house_test.drop(['price'],axis=1,inplace=True)
house_test.shape
house.shape
house_test.columns
newhouse = house_test.drop(['area_type','availability', 'location', 'size', 'society'],axis=1)
newhouse.head()
newhouse.isnull().sum()

#new_house_train = house.drop(['availability', 'location', 'size', 'society','price',
#                             'area_type','Electronic City Phase II'],axis=1)

#new_house_train.shape
#new_house_train.columns

#'Bannerghatta Road','Rajaji Nagar'
#'hoodi','jakkur'

newhouse.shape
house.shape

##  house.drop(['society','size','availability'],axis=1,inplace=True)

# EDA

house.head(10)
house.corr()['price'].sort_values(ascending=False)
sns.heatmap(house.corr(),annot=True)
sns.distplot(house['bath'],bins=5)
sns.pairplot(data=house,vars=['New_area_type', 'BHK','total_sqft', 'bath', 'balcony'],)
sns.lmplot(x='BHK',y='price',data=house)
sns.lmplot(x='bath',y='price',data=house)


# Train test split

X = house.drop(['availability', 'location', 'size', 'society','price','area_type'
                ,'Electronic City Phase II'],axis=1).values
y = house['price'].values

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train,y_train)

y_prediction = lin_reg.predict(X_test)
y_prediction

lin_reg.score(X_train,y_train)
lin_reg.score(X_test,y_test)
###################################################
RR = Ridge(alpha=50)
RR.fit(X_train,y_train)

Y_prediction2 = RR.predict(X_test)

RR.score(X_train,y_train)
RR.score(X_test,y_test)
#######################################################
K_r = KNeighborsRegressor(n_neighbors = 8)
K_r.fit(X_train, y_train)

Y_prediction3 = K_r.predict(X_test)
Y_prediction3
Y_prediction017 = K_r.predict(newhouse)
Y_prediction017

K_r.score(X_train,y_train)
K_r.score(X_test,y_test)

#######################################################

pred = pd.Series(Y_prediction017)
pred.head()
Submit017 = pd.DataFrame(columns=['price'])
Submit017.head()
Submit017['price'] = pred.astype(float)
Submit017.head()
Submit017.to_csv(r'Submit017.csv',index=False)

#######################################################
