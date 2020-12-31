import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
import lightgbm as lgb


pd.set_option('max_columns',None)
pd.set_option('max_rows',None)

HR_train = pd.read_csv('train_LZdllcl.csv')
HR_test = pd.read_csv('test_2umaH9m.csv')
HR_train.shape
HR_test.shape
HR_train.head()
HR_test.head()
HR_train.describe()
HR_train.info()
HR_train.nunique()

HR_train['is_promoted'].describe()
HR_train['is_promoted'].isnull().sum()
HR_train['is_promoted'].unique()
HR_train['is_promoted'].head()
Promoted = HR_train[HR_train['is_promoted']==1]
len(Promoted)
Not_Promoted = 54808 - 4668
Not_Promoted

print('Promotion rate % is',(len(Promoted)/(len(HR_train))) * 100)          ## 8.51
print('No Promotion rate % is',100 - (len(Promoted)/(len(HR_train))) * 100) ## 91.49
sns.distplot(HR_train['is_promoted'],color='blue')
pd.pivot_table(data=HR_train,values=['is_promoted'],columns=['education'],
               index=['no_of_trainings','age'],aggfunc=np.mean,margins=True,margins_name='Total')

HR_train['department'].describe()
HR_train['department'].unique()
HR_train['department'].isnull().any()
HR_train['department'].head()

HR_test['department'].describe()
len(HR_test['department'].unique())
HR_test['department'].isnull().any()
HR_test['department'].head()

HR_train['region'].describe()
HR_train['region'].unique()
HR_train['region'].isnull().any()
HR_train['region'].head()
HR_train['region'] = HR_train['region'].apply(lambda x:int(x.split('_')[1]))
sns.distplot(HR_train['region'])

HR_test['region'].describe()
HR_test['region'].isnull().any()
HR_test['region'].head()
HR_test['region'] = HR_test['region'].apply(lambda x:int(x.split('_')[1]))

HR_train['education'].describe()
HR_train['education'].unique()
HR_train['education'].isnull().any()
HR_train['education'].fillna(method='ffill',inplace=True)
HR_train['education'].head()

HR_test['education'].unique()
HR_test['education'].isnull().any()
HR_test['education'].fillna(method='ffill',inplace=True)

HR_train['gender'].describe()
HR_train['gender'].isnull().any()

HR_test['gender'].describe()
HR_test['gender'].isnull().any()

l1 = ['education','gender']
for New in l1:
    LE = LabelEncoder()
    HR_train[New] = LE.fit_transform(HR_train[New])

l2 = ['education','gender']
for New1 in l2:
    LE = LabelEncoder()
    HR_test[New1] = LE.fit_transform(HR_test[New1])

HR_train['education'].head()
HR_train['gender'].head()

HR_test['education'].head()
HR_test['gender'].head()

HR_train['recruitment_channel'].describe()
HR_train['recruitment_channel'].unique()
HR_train['recruitment_channel'].value_counts()
HR_train['recruitment_channel'].head()
pd.pivot_table(data=HR_train,values=['is_promoted'],index=['education','department'],columns=['recruitment_channel'])
HR_train['recruitment_channel'] = HR_train['recruitment_channel'].replace(['sourcing','other','referred'],
                                                                          [0,1,2])

HR_test['recruitment_channel'].describe()
HR_test['recruitment_channel'].head()
HR_test['recruitment_channel'] = HR_test['recruitment_channel'].replace(['sourcing','other','referred'],
                                                                          [0,1,2])

HR_train['no_of_trainings'].describe()
HR_train['no_of_trainings'].unique()
HR_train['no_of_trainings'].value_counts()
HR_train['no_of_trainings'].isna().any()
HR_train['no_of_trainings'].head()
sns.countplot(HR_train['no_of_trainings'])
sns.boxplot(HR_train['no_of_trainings'])
mm = MinMaxScaler()
HR_train['no_of_trainings'] = mm.fit_transform(HR_train[['no_of_trainings']])

HR_test['no_of_trainings'].describe()
HR_test['no_of_trainings'].isna().any()
HR_test['no_of_trainings'].head()

HR_train['age'].describe()
HR_train['age'].unique()
HR_train['age'].value_counts()
HR_train['age'].isna().any()
HR_train['age'].head()
sns.distplot(HR_train['age'])

HR_test['age'].describe()
HR_test['age'].isna().any()

HR_train['previous_year_rating'].describe()
HR_train['previous_year_rating'].unique()
HR_train['previous_year_rating'].value_counts()
HR_train['previous_year_rating'].isna().sum()
HR_train.groupby('previous_year_rating')['no_of_trainings'].value_counts().unstack()

def fill_previous_year_rating(data):
    previous_year_rating = data[0]
    no_of_trainings = data[1]

    if pd.isnull(previous_year_rating):
        if no_of_trainings == 1:
            return 3
        else:
            return 5
    else:
        return previous_year_rating

HR_train['previous_year_rating'] = HR_train[['previous_year_rating',
                                                         'no_of_trainings']].apply(fill_previous_year_rating,axis=1)
sns.countplot(HR_train['previous_year_rating'])

HR_test['previous_year_rating'].describe()
HR_test['previous_year_rating'].unique()
HR_test['previous_year_rating'].isna().sum()

def fill_previous_year_rating(data):
    previous_year_rating = data[0]
    no_of_trainings = data[1]

    if pd.isnull(previous_year_rating):
        if no_of_trainings == 1:
            return 3
        else:
            return 5
    else:
        return previous_year_rating

HR_test['previous_year_rating'] = HR_test[['previous_year_rating',
                                                         'no_of_trainings']].apply(fill_previous_year_rating,axis=1)



HR_train['length_of_service'].describe()
HR_train['length_of_service'].unique()
HR_train['length_of_service'].value_counts()
HR_train['length_of_service'].isna().sum()
sns.distplot(HR_train['length_of_service'])

HR_test['length_of_service'].describe()
HR_test['length_of_service'].unique()
HR_test['length_of_service'].isna().sum()

HR_train['KPIs_met >80%'].describe()
HR_train['KPIs_met >80%'].unique()
HR_train['KPIs_met >80%'].value_counts()
HR_train['KPIs_met >80%'].isna().sum()
sns.countplot(HR_train['KPIs_met >80%'])

HR_test['KPIs_met >80%'].describe()
HR_test['KPIs_met >80%'].unique()
HR_test['KPIs_met >80%'].isna().sum()

HR_train['awards_won?'].describe()
HR_train['awards_won?'].unique()
HR_train['awards_won?'].value_counts()
HR_train['awards_won?'].isna().sum()
sns.countplot(HR_train['awards_won?'])

HR_test['awards_won?'].describe()
HR_test['awards_won?'].isna().sum()

HR_train['avg_training_score'].describe()
HR_train['avg_training_score'].unique()
HR_train['avg_training_score'].value_counts()
HR_train['avg_training_score'].isna().sum()
sns.distplot(HR_train['avg_training_score'])

HR_test['avg_training_score'].describe()
HR_test['avg_training_score'].isna().sum()

HR_train.columns
HR_test.columns
HR_train.corr()['is_promoted'].sort_values(ascending=False)

# EDA

plt.figure(figsize=(12,8))
plt.subplot(221)
sns.countplot(x=HR_train['previous_year_rating'],hue='is_promoted',data=HR_train)
plt.subplot(222)
sns.countplot(x=HR_train['education'],data=HR_train,hue='is_promoted')
plt.subplot(223)
sns.countplot(x=HR_train['region'],hue='is_promoted',data=HR_train)
plt.subplot(224)
sns.countplot(x=HR_train['awards_won?'],hue='is_promoted',data=HR_train)
plt.tight_layout()
sns.countplot(x=HR_train['region'],data=HR_train,hue='gender')
sns.jointplot(x=HR_train['region'],y=HR_train['education'],data=HR_train)
sns.jointplot(x=HR_train['education'],y=HR_train['age'],data=HR_train)
sns.barplot(x=HR_train['is_promoted'],y=HR_train['gender'])
plt.figure(figsize=(12,8))
plt.subplot(121)
sns.scatterplot(x=HR_train['is_promoted'],y=HR_train['no_of_trainings'])
plt.subplot(122)
sns.scatterplot(x=HR_train['is_promoted'],y=HR_train['previous_year_rating'])
sns.scatterplot(HR_train['no_of_trainings'],HR_train['education'])
sns.scatterplot(HR_train['previous_year_rating'],HR_train['no_of_trainings'])

# Train and Test Split
X_train = HR_train.drop(['is_promoted','employee_id'],axis=1)
Y_train = HR_train['is_promoted']
X_test = HR_test.drop(['employee_id'],axis=1)

X_train_cat = X_train.select_dtypes(exclude=[np.number])
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_test_cat = X_test.select_dtypes(exclude=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])

oh = OneHotEncoder(handle_unknown='ignore')
X_train_oh = oh.fit_transform(X_train_cat).toarray()
X_test_oh = oh.fit_transform(X_test_cat).toarray()

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_numeric)
X_test_sc = sc.transform(X_test_numeric)

X_train_final = np.hstack((X_train_oh,X_train_sc))
X_test_final = np.hstack((X_test_oh,X_test_sc))

X_train, X_test, y_train, y_test = train_test_split(X_train_final, Y_train, test_size=0.33, random_state=42)

log = LogisticRegression()
log.fit(X_train_final,Y_train)
Prediction = log.predict(X_test_final)
Prediction = pd.DataFrame(Prediction)
Prediction.head()
print('train score:',log.score(X_train_final,Y_train))
print('test score:',log.score(X_test_final,Prediction))

knn = KNeighborsClassifier()
knn.fit(X_train_final,Y_train)
Prediction01 = knn.predict(X_test_final)
print('train score:',knn.score(X_train_final,Y_train))
print('test score:',knn.score(X_test_final,Prediction01))
f1_scores = cross_val_score(knn,X_train_final,Y_train, cv=5, scoring='f1')
print("F1_score = ",f1_scores," Mean F1 score = ", np.mean(f1_scores))

gbm = lgb.LGBMClassifier(objective='binary')
gbm.fit(X_train_final,Y_train)
Prediction02 = gbm.predict(X_test_final)
print('train score:',gbm.score(X_train_final,Y_train))
print('test score:',gbm.score(X_test_final,Prediction02))
f1_scores = cross_val_score(gbm,X_train_final,Y_train, cv=5, scoring='f1',n_jobs=-1)
print("F1_score = ",f1_scores," Mean F1 score = ", np.mean(f1_scores))

submission01 = pd.DataFrame({'employee_id':HR_test['employee_id'],'is_promoted':Prediction02})
submission01.head()
submission01.to_csv("gbm01.csv", index=False)