#!/usr/bin/env python
# coding: utf-8

# <img src="https://miro.medium.com/max/450/0*KYIdwOJU-OGNX7vG" alt="edWisor">

# # PROJECT : CAB FARE PREDICTION

# ### Candidate name : Kuldeep Sharma.R

# ### Problem Statement -
#               You are a cab rental start-up company. You have successfully run the pilot project and now want to launch your cab service across the country.You have collected the historical data from your pilot project and now have a requirement to apply analytics for fare prediction. You need to design a system that predicts the fare amount for a cab ride in the city.

# >### Number of attributes:
# 1. **Pickup_datetime**    -timestamp value indicating when the cab ride started.
# 2. **Pickup_longitude**   - float for longitude coordinate of where the cab ride started.
# 3. **Pickup_latitude**    - float for latitude coordinate of where the cab ride started.
# 4. **Dropoff_longitude**  - float for longitude coordinate of where the cab ride ended.
# 5. **Dropoff_latitude**   - float for latitude coordinate of where the cab ride ended.
# 6. **Passenger_count**    - an integer indicating the number of passengers in the cab ride.

# ### Importing the File

# In[1]:


import pandas as pd

df = pd.read_csv("train_cab.csv")
df.head()


# In[2]:


df_test=pd.read_csv("test.csv")
df_test.head()


# In[3]:


print(df.info)
print(df_test.info)


# In[4]:


df.describe()


# In[5]:


df_test.describe()


# ## Feature Engineering

# ### Transforming the Date & time of the customer pickup datetime to usable case

# In[6]:


import datetime


# In[7]:


pd.to_datetime(df['pickup_datetime'],errors='coerce')


# In[8]:


df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],errors='coerce')


# In[9]:


df['pickup_datetime'].dt.month


# ##### Creating usable features using datetime data

# In[10]:


df['year']=df['pickup_datetime'].dt.year
df['month']=df['pickup_datetime'].dt.month
df['day']=df['pickup_datetime'].dt.day
df['hour']=df['pickup_datetime'].dt.hour
df['minute']=df['pickup_datetime'].dt.minute


# In[11]:


df.head()


# In[12]:


import numpy as np


# In[13]:


##Taking shift 0&1 as the AM & PM respectively.
df['shift']=np.where(df['hour']<12,0,1)


# In[14]:


df.drop('pickup_datetime',axis=1,inplace=True)
df.head()


# ### Calculating the distance travelled by the cab using Haversince distance method, Since geograhical distance is given

# In[15]:


# for distance travelled by cab, Calculating haversine distance

def haversine(df):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1= np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    #### Based on the formula  x1=drop_lat,x2=dropoff_long 
    dlat = np.radians(df['dropoff_latitude']-df["pickup_latitude"])
    dlong = np.radians(df["dropoff_longitude"]-df["pickup_longitude"])
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 6371 # Radius of earth in kilometers.
    return c * r


# In[16]:


df['total_distance']=haversine(df)


# In[17]:


df.head()


# In[18]:


# Since the Distance is calculated, latitude& longitude values are not required, son dropping them

df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)


# In[19]:


df.head()


# In[20]:


# Doing the Same for test data set

# Transforming to datetime
df_test['pickup_datetime']=pd.to_datetime(df_test['pickup_datetime'],errors='coerce')

# Creating usable features from datetime
df_test['year']=df_test['pickup_datetime'].dt.year
df_test['month']=df_test['pickup_datetime'].dt.month
df_test['day']=df_test['pickup_datetime'].dt.day
df_test['hour']=df_test['pickup_datetime'].dt.hour
df_test['minute']=df_test['pickup_datetime'].dt.minute

# Since Cab availability depends on time of the day, creating a feature to navigate AM & PM as 0 & 1 respectively
df_test['shift']=np.where(df_test['hour']<12,0,1)

# Applying the haversine distance method to calculate distance of cab travelled using its geographical locations
df_test['total_distance']=haversine(df_test)

# Dropping the features which are not gonna be in use from here
df_test.drop(['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)

df_test.head()


# In[21]:


print(df.dtypes)
df_test.dtypes


# In[22]:


# Since data type of fare_amount given as object which can't be, so converting it into float

df["fare_amount"]=pd.to_numeric(df["fare_amount"],errors='coerce')
df["fare_amount"].dtype


# In[23]:


df.shape


# ## Exploratory Data Analysis

# ### Checking missing values & Handling them

# In[24]:


df.isnull().sum()


# In[25]:


variable=['fare_amount']#,'passenger_count','year','month','day','hour','minute']
for median in df[variable]:
    print(df[variable].median())


# In[26]:


df.median()


# In[27]:


df['fare_amount'].fillna(df['fare_amount'].median(),inplace=True)
df['passenger_count'].fillna(df['passenger_count'].median(),inplace=True)
df['year'].fillna(df['year'].median(),inplace=True)
df['month'].fillna(df['month'].median(),inplace=True)
df['day'].fillna(df['day'].median(),inplace=True)
df['hour'].fillna(df['hour'].median(),inplace=True)
df['minute'].fillna(df['minute'].median(),inplace=True)


# In[28]:


print(df.shape)
df.isnull().sum()


# **Finding No.of Passengers Rode the cab**

# In[29]:


#Since the passenger_count cant be in Decimal
df["passenger_count"]=np.ceil(df["passenger_count"])
df["passenger_count"].value_counts()


# ### Since In a Cab Passengers can't be more than 6 people, Finding Passengers more than 6 people

# In[30]:


(df["passenger_count"]>6).value_counts()


# ### Replacing these Outlier values greater than 6 passenger_count with median

# In[31]:


df.drop(df[df['passenger_count']>6].index,inplace=True)


# In[32]:


df.shape


# In[33]:


df.head()


# ### Checking if the Fare amount is less than "Zero" - which is not possible

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


plt.figure(figsize=(20,20))
plt.subplot(321)
sns.distplot(df['fare_amount'],bins=50)
plt.subplot(322)
sns.distplot(df['total_distance'],bins=50)
plt.subplot(323)
sns.distplot(df['passenger_count'],bins=50)
plt.subplot(324)
sns.distplot(df['year'],bins=50)
# plt.savefig('hist.png')
plt.show()


# ### The Distribution is Right skewed with outliers present in it, therefore lets handle the outliers

# ##### --> Handling outliers in Fare amount of the cab

# ##### Copying original dataset to perform outlier treatment (This will we useful for Linear  models & Neural Networks)

# In[36]:


df1=df.copy()
df1.head()


# In[37]:


df1.fare_amount.describe()


# ##### Finding InterQuantile range for fare amount charged for passengers

# In[38]:


# Calculating InterQuantile range to know boundaries
IQR=df1.fare_amount.quantile(0.75)-df1.fare_amount.quantile(0.25)
print('InterQuantile Range  : {}'.format(IQR))

# computing outlier
upper_bridge_fare=df1.fare_amount.quantile(0.75)+(1.5*IQR)
lower_bridge_fare=df1.fare_amount.quantile(0.25)-(1.5*IQR)
print('upper bridge         : {}'.format(upper_bridge_fare))
print('lower bridge         : {}'.format(lower_bridge_fare))

# Computing Extreme Outlier
ext_upper_bridge_fare=df1.fare_amount.quantile(0.75)+(3*IQR)
ext_lower_bridge_fare=df1.fare_amount.quantile(0.25)-(3*IQR)
print('Extreme upper bridge : {}'.format(ext_upper_bridge_fare))
print('Extreme lower bridge : {}'.format(ext_lower_bridge_fare))


# ### Since the Extreme lower bridge for fare amount charged for passenger is 32, considering all other values greater than that as outlier

# In[39]:


print((df1['fare_amount']>ext_upper_bridge_fare).value_counts())

# Also the fare charged cant be negative
(df1['fare_amount']<0).value_counts()


# In[40]:


# df1.drop(df1[df1['fare_amount']>ext_upper_bridge_fare].index,inplace=True)
df1.loc[df1['fare_amount']>ext_upper_bridge_fare,'fare_amount']=ext_upper_bridge_fare
df1.drop(df1[df1['fare_amount']<0].index,inplace=True)
# df1.loc[df['fare_amount']>=32,'fare_amount']=32

# Re-checking the whether fare amount is greater than 32
sns.boxplot(df1['fare_amount'])


# In[41]:


(df1['fare_amount']>ext_upper_bridge_fare).value_counts()


# In[42]:


# Calculating InterQuantile range to know boundaries
IQR=df1.total_distance.quantile(0.75)-df.total_distance.quantile(0.25)
print('InterQuantile Range  : {}'.format(IQR))

# computing outlier
upper_bridge_dist=df1.total_distance.quantile(0.75)+(1.5*IQR)
lower_bridge_dist=df1.total_distance.quantile(0.25)-(1.5*IQR)
print('upper bridge         : {}'.format(upper_bridge_dist))
print('lower bridge         : {}'.format(lower_bridge_dist))

# Computing Extreme Outlier
ext_upper_bridge_dist=df1.total_distance.quantile(0.75)+(3*IQR)
ext_lower_bridge_dist=df1.total_distance.quantile(0.25)-(3*IQR)
print('Extreme upper bridge : {}'.format(ext_upper_bridge_dist))
print('Extreme lower bridge : {}'.format(ext_lower_bridge_dist))


# ### Choosing The Extreme outlier for the Outliers

# In[43]:


# Since the Extreme boundries with IQR provides a limit after which if values located we can consider as Outlier
print((df1['total_distance']>ext_upper_bridge_dist).value_counts())

# Also the distance charged cant be negative
(df1['total_distance']<0).value_counts()


# In[44]:


# df1.drop(df1[df1['total_distance']>ext_upper_bridge_dist].index,inplace=True)

df1.loc[df1['total_distance']>ext_upper_bridge_dist,'total_distance']=ext_upper_bridge_dist
# df.drop(df[df['total_distance']<0].index,inplace=True)

# Re-checking the whether fare amount is greater than 32
sns.boxplot(df1['total_distance'])


# In[45]:


(df1['total_distance']>ext_upper_bridge_dist).value_counts()


# ### Correlation

# In[46]:


sns.pairplot(df1)


# In[47]:


# Get correlation o0f each features in dataset
corrmat = df1.corr()
print(corrmat.index)
corrmat


# ### Correlation with Heat Map

# In[48]:


top_corr_features = corrmat.index

plt.figure(figsize=(20,10))

# Plot Heatmap
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[49]:


df1.shape


# ## Feature Importance

# In[50]:


X=df1.iloc[:,1:] # Independent Features
y=df1.iloc[:,0]  # Dependent Features


# In[51]:


# Feature Importance using ExtraTreesRegressor Module
from sklearn.ensemble import ExtraTreesRegressor


# In[52]:


select_feature=ExtraTreesRegressor()
select_feature.fit(X,y)


# In[53]:


select_feature.feature_importances_


# In[54]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(select_feature.feature_importances_, index=X.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()


# ## Machine Learning

# ### Train-Test-split for train & test dataset

# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


# In[56]:


plt.figure(figsize=(20,10))
sns.distplot(y)


# ### Linear Regression

# In[57]:


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)


# In[58]:


print('Coefficient of determination R^2 - on train set = {}'.format(lin_reg.score(X_train,y_train)))
print('Coefficient of determination R^2 - on test set = {}'.format(lin_reg.score(X_test,y_test)))


# In[59]:


from sklearn.model_selection import cross_val_score
lin_score=cross_val_score(lin_reg,X,y,cv=10)


# In[60]:


lin_score.mean()


# ##### Model visualization for prediction pattern

# In[61]:


linear_regression_model=lin_reg.predict(X_test)
sns.distplot(y_test-linear_regression_model)


# In[62]:


plt.scatter(y_test,linear_regression_model)


# - The model has performed well as the Distribution now is Normal & points are scatter to form a linear line is visible

# ##### Evaluation Metrics

# In[63]:


from sklearn import metrics
print('MAE : {}'.format(metrics.mean_absolute_error(y_test,linear_regression_model)))
print('MSE : {}'.format(metrics.mean_squared_error(y_test,linear_regression_model)))
print('RMSE : {}'.format(np.sqrt(metrics.mean_absolute_error(y_test,linear_regression_model))))


# ##### Dumping file to pickle for Model deployment

# In[64]:


import pickle

# #open a folder where pickle file to be stored
file = open('lin_reg_model.pkl','wb')

# # dum informations to that file
pickle.dump(linear_regression_model,file)


# In[65]:


# Initially lets check best score occuring from Linear regression

linear_regression_model=LinearRegression()
mse=cross_val_score(lin_reg,X,y,scoring='neg_root_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# ### Ridge Regression

# In[67]:


# Now let's see whether Ridge Regression performs better than Linear
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge_reg=Ridge()
ridge_score=cross_val_score(ridge_reg,X,y,cv=10)
print('Ridge score by CV              :{}'.format(ridge_score.mean()))

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,100]}
ridge_regressor=GridSearchCV(ridge_reg,parameters,scoring='neg_root_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)

print('Best parameter by GridSearchCV : {}'.format(ridge_regressor.best_params_))
print('Best score by Neg_RSME         : {}'.format(ridge_regressor.best_score_))


# ### Lasso Regression

# In[68]:


# Let's check with Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso_reg=Lasso()
lasso_score=cross_val_score(lasso_reg,X,y,cv=10)
print('Lasso score by CV              :{}'.format(lasso_score.mean()))

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55]}
lasso_regressor=GridSearchCV(lasso_reg,parameters,scoring='neg_root_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print('Best parameter by GridSearchCV : {}'.format(lasso_regressor.best_params_))
print('Best score by Neg_RSME         : {}'.format(lasso_regressor.best_score_))


# - There is no Noticeable Changes in the best score performend by Ridge & Lasso when compared with Linear Regression, we might get better results when used with hige dataset    (So,I'm not gonna dump these in pickle model)

# ### Decision Tree Classifier

# In[69]:


from sklearn.tree import DecisionTreeRegressor

dtree=DecisionTreeRegressor()
dtree.fit(X_train,y_train)


# In[70]:


print("Coefficient of determination R^2 <-- on train set: {}".format(dtree.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtree.score(X_test, y_test)))


# In[71]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(dtree,X,y,cv=10)
score.mean()


# ##### Model visualization for prediction pattern

# In[72]:


dtree_prediction=dtree.predict(X_test)
sns.distplot(y_test-dtree_prediction)


# In[73]:


plt.scatter(y_test,dtree_prediction)


# In[74]:


print('MAE:', metrics.mean_absolute_error(y_test, dtree_prediction))
print('MSE:', metrics.mean_squared_error(y_test, dtree_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtree_prediction)))


# ##### Hyperparameter Tuning for DecisionTreeRegressor

# In[75]:


params={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3],
"min_weight_fraction_leaf":[0.1,0.2],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[30,40,50,60,70]
    }


# In[76]:


from sklearn.model_selection import GridSearchCV
grid_search_dtree=GridSearchCV(dtree,param_grid=params,scoring='neg_root_mean_squared_error',n_jobs=-1,cv=5,verbose=3)


# In[77]:


grid_search_dtree.fit(X,y)


# In[78]:


print(grid_search_dtree.best_params_)
print(grid_search_dtree.best_score_)


# In[79]:


dtree_predictions=grid_search_dtree.predict(X_test)
sns.distplot(y_test-dtree_predictions)


# ##### Evaluation Metrics.

# In[80]:


print('MAE:', metrics.mean_absolute_error(y_test, dtree_predictions))
print('MSE:', metrics.mean_squared_error(y_test, dtree_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtree_predictions)))


# - Decision tree has not performed better than Linear regession

# ##### Dumping file to pickle for Model deployment

# In[82]:


# # open a file, where you ant to store the data
file = open('decision_regression_model.pkl', 'wb')

# # dump information to that file
pickle.dump(grid_search_dtree, file)


# ### RandomForest Regression

# In[83]:


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()
rf_reg.fit(X,y)


# In[84]:


print("Coefficient of determination R^2 <-- on train set: {}".format(rf_reg.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(rf_reg.score(X_test, y_test)))


# In[85]:


from sklearn.model_selection import cross_val_score
rf_score=cross_val_score(rf_reg,X,y,cv=5)
rf_score.mean()


# ##### Model visualization for prediction pattern

# In[86]:


rf_prediction=rf_reg.predict(X_test)
sns.distplot(y_test-rf_prediction)


# In[87]:


plt.scatter(y_test,rf_prediction)


# In[88]:


print('MAE:', metrics.mean_absolute_error(y_test, rf_prediction))
print('MSE:', metrics.mean_squared_error(y_test, rf_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_prediction)))


# In[89]:


from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[90]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
rf_random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(rf_random_grid)


# In[91]:


rf_random = RandomizedSearchCV(estimator = rf_reg, param_distributions = rf_random_grid,scoring='neg_root_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[92]:


rf_random.fit(X_train,y_train)


# In[93]:


print(rf_random.best_params_)
print(rf_random.best_score_)


# In[94]:


rf_predictions=rf_random.predict(X_test)
sns.distplot(y_test-rf_predictions)


# ##### Evaluation Metrics

# In[95]:


print('MAE :', metrics.mean_absolute_error(y_test, rf_predictions))
print('MSE :', metrics.mean_squared_error(y_test, rf_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_predictions)))


# ##### Dumping file to pickle for Model deployment

# In[96]:


# # open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# # dump information to that file
pickle.dump(rf_random, file)


# #### XGBoost

# In[97]:


import xgboost as xgb
xgb_regressor=xgb.XGBRegressor()
xgb_regressor.fit(X_train,y_train)


# In[98]:


print("Coefficient of determination R^2 <-- on train set: {}".format(xgb_regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(xgb_regressor.score(X_test, y_test)))


# In[99]:


from sklearn.model_selection import cross_val_score
xgb_score=cross_val_score(xgb_regressor,X,y,cv=5)
xgb_score.mean()


# ##### Model visualization for prediction pattern

# In[100]:


xgb_prediction=xgb_regressor.predict(X_test)
sns.distplot(y_test-xgb_prediction)


# In[101]:


plt.scatter(y_test,xgb_prediction)


# ##### Evaluation Metrics

# In[102]:


print('MAE:', metrics.mean_absolute_error(y_test, xgb_prediction))
print('MSE:', metrics.mean_squared_error(y_test, xgb_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_prediction)))


# In[103]:


xgb.XGBRegressor()


# In[104]:


# Parameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]

# Creating the random grid
xgb_random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

print(xgb_random_grid)


# In[105]:


xg_random = RandomizedSearchCV(estimator = xgb_regressor, param_distributions = xgb_random_grid,scoring='neg_root_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[106]:


xg_random.fit(X_train,y_train)


# In[107]:


print(xg_random.best_params_)
print(xg_random.best_score_)


# In[108]:


xgb_predictions=xg_random.predict(X_test)


# In[109]:


sns.distplot(y_test-xgb_predictions)


# ##### Evaluation Metrics

# In[110]:


print('MAE:', metrics.mean_absolute_error(y_test, xgb_predictions))
print('MSE:', metrics.mean_squared_error(y_test, xgb_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_predictions)))


# ##### Dumping file to pickle for Model deployment

# In[111]:


# # open a file, where you ant to store the data
file = open('xg_boost model.pkl', 'wb')

# # dump information to that file
pickle.dump(xg_random, file)


# ## Best Model for prediction

# ### Model Scores
# 
# 1. Linear Regression         :   0.7424330384503302
# 2. Ridge Regression          :   0.7424330590469321
# 3. Lasso Regression          :   0.7327516901466671
# 4. Decision Trees Regression :   0.5205541179521329
# 5. Random Forest Regression  :   **0.7617557962178525**
# 6. XgBoost Regression        :   0.7552525126745274
# 
# It is clear that ***Random Forest Regression*** is Best Performing Model

# ### Negative RSME from cross validation
# 
# 1. Linear Regression         :   -3.581147977164347
# 2. Ridge Regression          :   -3.5811334699139517
# 3. Lasso Regression          :   -3.5811479771432557
# 4. Decision Trees Regression :   -4.0718292988961355
# 5. Random Forest Regression  :   **-3.3087780047498954**
# 6. XgBoost Regression        :   -3.6252214480452567
# 
# It is clear that here too ***RandomForest Regression*** is best performing Model

# Therefore, Performing Random Forest Regression on the Test DataSet with feature engineering is as follows

# In[112]:


df_test.head()


# ## Prediction for Test Data Set

# In[113]:


df_test['fare_amount_predicted'] = rf_random.predict(df_test)


# In[114]:


df_test.head()


# In[115]:


df_test.fare_amount_predicted.describe()


# ### Exporting the CSV as predicted_cab_fare

# In[116]:


df_test.to_csv('predicted_cab_fare.csv',index=False)


# In[66]:


# df.to_csv('train_data.csv',index=False)

