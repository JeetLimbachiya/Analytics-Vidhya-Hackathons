# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,explained_variance_score
from openpyxl import Workbook

pd.set_option("max_rows",None)
pd.set_option("max_columns",None)

# 2. Read the file

f = pd.read_excel("Flight_Data_Train.xlsx")
flight1 = pd.read_excel("Flight_Test_set.xlsx")

# 3. Understand the data

f.info()
flight1.info()
f.describe()
flight1.describe()
f.shape

#######
f['Airline'].unique()
flight1['Airline'].unique()
f['Airline'].value_counts()
sns.boxplot(x='Airline',y='Price',data=f)
Airline = f['Airline']    ##### Here because it is nominal and not order data
Airline1 = flight1['Airline']
Airline = pd.get_dummies(Airline,drop_first=True)
Airline1 = pd.get_dummies(Airline1,drop_first=True)
Airline.drop(['Trujet'],inplace=True,axis=1)
Airline.shape
Airline1.shape
Airline.head()
Airline1.head()

#######
f['Source'].unique()
flight1['Source'].unique()
f['Source'].value_counts()
Source = f['Source']    ##### Here because it is nominal and not order data
Source1 = flight1['Source']
Source = pd.get_dummies(Source,drop_first=True)
Source1 = pd.get_dummies(Source1,drop_first=True)
Source.shape
Source1.shape
Source.head()
Source1.head()
#######
f['Destination'].unique()
f['Destination'].value_counts()
Destination = f['Destination']    ##### Here because it is nominal and not order data
Destination1 = flight1['Destination']
Destination = pd.get_dummies(Destination,drop_first=True)
Destination1 = pd.get_dummies(Destination1,drop_first=True)
Destination.head()
Destination1.head()
Destination.dtypes

f.columns
f.dtypes
#######

#######
f['Total_Stops'].describe()
f['Total_Stops'].unique()
flight1['Total_Stops'].unique()
f['Total_Stops'].value_counts()
f['Total_Stops'].isnull().sum()
flight1['Total_Stops'].isnull().sum()
f['Total_Stops'].fillna(value='1 stop',inplace=True)
f['Total_Stops'].isnull().sum()
le2 = LabelEncoder()
flight1['Total_Stops'] = le2.fit_transform(flight1['Total_Stops'])
flight1['Total_Stops'].values
flight1['Total_Stops'].describe()
le1 = LabelEncoder()                ## Here order like 1, 2, 3
f['Total_Stops'] = le1.fit_transform(f['Total_Stops'])
f['Total_Stops'].values
f['Total_Stops'].describe()
sc = StandardScaler()
f['Total_Stops'] = sc.fit_transform(f[['Total_Stops']])
f['Total_Stops'].describe()
f['Total_Stops'].head()
sc1 = StandardScaler()
flight1['Total_Stops'] = sc1.fit_transform(flight1[['Total_Stops']])
flight1['Total_Stops'].describe()

sns.boxplot(f['Total_Stops'],f['Price'],data=f)
f['Total_Stops'].min()
f['Total_Stops'].max()
f['Total_Stops'] = f['Total_Stops'].apply(lambda x: (x-f['Total_Stops'].min())/(f['Total_Stops'].max()-f['Total_Stops'].min()))
sns.boxplot(f['Total_Stops'],f['Price'],data=f)
flight1['Total_Stops'] = flight1['Total_Stops'].apply(lambda x: (x-flight1['Total_Stops'].min())/(flight1['Total_Stops'].max()-flight1['Total_Stops'].min()))
f['Total_Stops'].describe()
f['Total_Stops'].shape
flight1['Total_Stops'].shape
flight1['Total_Stops'].describe()
########
f["Date_of_Journey"] = pd.to_datetime(f["Date_of_Journey"])
f["Date_of_Journey"].dtype
f["Day"] =f["Date_of_Journey"].dt.day
f["Month"] =f["Date_of_Journey"].dt.month
f.head()
f.drop('Date_of_Journey',axis=1,inplace=True)
f.columns

flight1["Date_of_Journey"] = pd.to_datetime(flight1["Date_of_Journey"])
flight1["Date_of_Journey"].dtype
flight1["Day"] =flight1["Date_of_Journey"].dt.day
flight1["Month"] =flight1["Date_of_Journey"].dt.month
flight1.drop('Date_of_Journey',axis=1,inplace=True)
flight1.columns

f["Day"].describe()
f["Day"] = f["Day"].apply(lambda x: (x-f["Day"].min())/(f["Day"].max()-f["Day"].min()))
f["Day"].describe()
f["Month"].describe()
f["Month"] = f["Month"].apply(lambda x: (x-f["Month"].min())/(f["Month"].max()-f["Month"].min()))
f["Month"].describe()

flight1["Day"].describe()
flight1["Day"] = flight1["Day"].apply(lambda x: (x-flight1["Day"].min())/(flight1["Day"].max()-flight1["Day"].min()))
flight1["Month"].describe()
flight1["Month"] = flight1["Month"].apply(lambda x: (x-flight1["Month"].min())/(flight1["Month"].max()-flight1["Month"].min()))

##########
f['Dep_Time'].describe()
f['Dep_Time'] = pd.to_datetime(f['Dep_Time'])
f['Dep_Time'].dtype
f["Dept_Hour"] =f['Dep_Time'].dt.hour
f["Dept_Min"] =f['Dep_Time'].dt.minute
f.head()
f.drop('Dep_Time',axis=1,inplace=True)

flight1['Dep_Time'].describe()
flight1['Dep_Time'] = pd.to_datetime(flight1['Dep_Time'])
flight1['Dep_Time'].dtype
flight1["Dept_Hour"] =flight1['Dep_Time'].dt.hour
flight1["Dept_Min"] =flight1['Dep_Time'].dt.minute
flight1.head()
flight1.drop('Dep_Time',axis=1,inplace=True)

f["Dept_Hour"].describe()
f["Dept_Hour"] = f["Dept_Hour"].apply(lambda x: (x-f["Dept_Hour"].min())/(f["Dept_Hour"].max()-f["Dept_Hour"].min()))
f["Dept_Hour"].describe()
f["Dept_Min"].describe()
f["Dept_Min"] = f["Dept_Min"].apply(lambda x: (x-f["Dept_Min"].min())/(f["Dept_Min"].max()-f["Dept_Min"].min()))
f["Dept_Min"].describe()

flight1["Dept_Hour"].describe()
flight1["Dept_Hour"] = flight1["Dept_Hour"].apply(lambda x: (x-flight1["Dept_Hour"].min())/(flight1["Dept_Hour"].max()-flight1["Dept_Hour"].min()))
flight1["Dept_Hour"].describe()
flight1["Dept_Min"].describe()
flight1["Dept_Min"] = flight1["Dept_Min"].apply(lambda x: (x-flight1["Dept_Min"].min())/(flight1["Dept_Min"].max()-flight1["Dept_Min"].min()))
flight1["Dept_Min"].describe()

f.columns
########
f['Arrival_Time'].describe()
f['Arrival_Time'] = pd.to_datetime(f['Arrival_Time'])
f['Arrival_Time'].dtype
f["Arr_Hour"] =f['Arrival_Time'].dt.hour
f["Arr_Min"] =f['Arrival_Time'].dt.minute
f.head()
f.drop('Arrival_Time',axis=1,inplace=True)

flight1['Arrival_Time'].describe()
flight1['Arrival_Time'] = pd.to_datetime(flight1['Arrival_Time'])
flight1['Arrival_Time'].dtype
flight1["Arr_Hour"] =flight1['Arrival_Time'].dt.hour
flight1["Arr_Min"] =flight1['Arrival_Time'].dt.minute
flight1.drop('Arrival_Time',axis=1,inplace=True)

f["Arr_Hour"].describe()
f["Arr_Hour"] = f["Arr_Hour"].apply(lambda x: (x-f["Arr_Hour"].min())/(f["Arr_Hour"].max()-f["Arr_Hour"].min()))
f["Arr_Hour"].describe()
f["Arr_Min"].describe()
f["Arr_Min"] = f["Arr_Min"].apply(lambda x: (x-f["Arr_Min"].min())/(f["Arr_Min"].max()-f["Arr_Min"].min()))
f["Arr_Min"].describe()

flight1["Arr_Hour"].describe()
flight1["Arr_Hour"] = flight1["Arr_Hour"].apply(lambda x: (x-flight1["Arr_Hour"].min())/(flight1["Arr_Hour"].max()-flight1["Arr_Hour"].min()))
flight1["Arr_Hour"].describe()
flight1["Arr_Min"].describe()
flight1["Arr_Min"] = flight1["Arr_Min"].apply(lambda x: (x-flight1["Arr_Min"].min())/(flight1["Arr_Min"].max()-flight1["Arr_Min"].min()))
flight1["Arr_Min"].describe()
flight1.columns

f.columns
######
f['Source'].unique()
f['Source'].value_counts()
f['Source'].head(2)
f.head(2)
#######
f['Additional_Info'].describe()
f['Additional_Info'].unique()
f['Additional_Info'] = f['Additional_Info'].replace(['No info'],0)
f['Additional_Info'] = f['Additional_Info'].replace(['In-flight meal not included'],1)
f['Additional_Info'] = f['Additional_Info'].replace(['No check-in baggage included'],2)
f['Additional_Info'] = f['Additional_Info'].replace(['1 Short layover'],3)
f['Additional_Info'] = f['Additional_Info'].replace(['No Info'],4)
f['Additional_Info'] = f['Additional_Info'].replace(['1 Long layover'],5)
f['Additional_Info'] = f['Additional_Info'].replace(['Change airports'],6)
f['Additional_Info'] = f['Additional_Info'].replace(['Business class'],7)
f['Additional_Info'] = f['Additional_Info'].replace(['Red-eye flight'],8)
f['Additional_Info'] = f['Additional_Info'].replace(['2 Long layover'],9)
f['Additional_Info'].unique()
f['Additional_Info'].describe()
f['Additional_Info'] = f['Additional_Info'].apply(lambda x: (x-f['Additional_Info'].min())/(f['Additional_Info'].max()-f['Additional_Info'].min()))
f['Additional_Info'].describe()
sns.boxplot(f['Additional_Info'],f['Price'],data=f)
f.columns

flight1['Additional_Info'].describe()
flight1['Additional_Info'].unique()
flight1['Additional_Info'] = flight1['Additional_Info'].replace(['No info'],0)
flight1['Additional_Info'] = flight1['Additional_Info'].replace(['In-flight meal not included'],1)
flight1['Additional_Info'] = flight1['Additional_Info'].replace(['No check-in baggage included'],2)
flight1['Additional_Info'] = flight1['Additional_Info'].replace(['1 Long layover'],3)
flight1['Additional_Info'] = flight1['Additional_Info'].replace(['Change airports'],5)
flight1['Additional_Info'] = flight1['Additional_Info'].replace(['Business class'],4)
flight1['Additional_Info'].unique()
flight1['Additional_Info'] = flight1['Additional_Info'].apply(lambda x: (x-flight1['Additional_Info'].min())/(flight1['Additional_Info'].max()-flight1['Additional_Info'].min()))
flight1['Additional_Info'].describe()

#######
Train_Flight = pd.concat([f,Airline,Source,Destination],axis=1)
Train_Flight.head(2)
Train_Flight.drop(['Airline','Source','Destination'],axis=1,inplace=True)
Train_Flight.head(2)
Train_Flight.shape
Train_Flight.isnull().sum()
Train_Flight.drop('Route',axis=1,inplace=True)
Train_Flight.drop('Duration',axis=1,inplace=True)
Train_Flight.shape
Train_Flight.isnull().sum()
Train_Flight.columns

flight1.columns
Test_Flight1 = pd.concat([flight1,Airline1,Source1,Destination1],axis=1)
Test_Flight1.head(2)
Test_Flight1.drop(['Airline','Source','Destination'],axis=1,inplace=True)
Test_Flight1.head(2)
Test_Flight1.shape
Test_Flight1.isnull().sum()
Test_Flight1.drop('Route',axis=1,inplace=True)
Test_Flight1.drop('Duration',axis=1,inplace=True)
Test_Flight1.columns

#######

Train_Flight.corr()['Price'].sort_values(ascending=False)

######
# EDA
sns.catplot(x="Airline",y="Price",data=f)  ### Because we have category also
sns.catplot(x="Source",y="Price",data=f)  ### Because we have category also
sns.boxplot(x="Airline",y="Price",data=f)  ### Because we have category also
sns.boxplot(x="Source",y="Price",data=f) ### Because we have category also
sns.countplot(x='Additional_Info',data=f)
sns.pairplot(Train_Flight,vars=['Price','Additional_Info','Month','Arr_Hour'])

plt.figure(figsize=(12,12))
sns.heatmap(Train_Flight.corr(),annot=True,fmt=".1g",vmax=.8,square=True)
plt.ylim(0,31)
#####

#### Test Train Split

Train_Flight.columns
X = Train_Flight.drop(['Price'],axis=1).values
y = Train_Flight['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#### LASSO REGRESSION ############################################
lasso = Lasso(alpha=500)
lasso.fit(X_train,y_train)

y_predict = lasso.predict(X_test)
lasso.score(X_train,y_train)
lasso.score(X_test,y_test)

plt.scatter(y_test, y_predict, color='purple')
plt.xlabel = ('Actual y')
plt.ylabel = ('Predicted y')
plt.title = ('Actual y vs Predicted y')

### Metrics

n = len(X_test)
k = X_test.shape[1]  # independent parameter
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
r2
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
adj_r2

sns.distplot(y_test-y_predict)
plt.show()
sns.scatterplot(y_test,y_predict,alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_predict")
plt.show()
##################################################################

############### Linear Regression ################################################
X = Train_Flight.drop(['Price'],axis=1).values
y = Train_Flight['Price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

L_R = LinearRegression(fit_intercept=True)   # By applying lambda
L_R.fit(X_train,y_train)

y_predict1 = L_R.predict(X_test)               
y_predict1 = L_R.predict(X_train)
L_R.score(X_train,y_train)
L_R.score(X_test,y_test)

y_predict12 = L_R.predict(Test_Flight1)
y_predict12.shape
#######################################################################################
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train,y_train)


Y_prediction0 = reg_rf.predict(X_test)
Y_prediction2468 = reg_rf.predict(Test_Flight1)
Y_prediction2468
Y_prediction2468.shape
reg_rf.score(X_train,y_train)*100

#######################################################################################
Y_prediction2468

Prediction10 = pd.Series(Y_prediction2468)
Prediction10.head()

Submit10 = pd.DataFrame(columns=['Price'])
Submit10.head()
Submit10['Price'] = Prediction10.astype(int)
Submit10.head()
Submit10.to_excel(r'Submit10.xlsx',index=False)
###############################################################


