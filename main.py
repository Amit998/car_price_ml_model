from numpy.core.function_base import linspace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('car data.csv')

# print(df.head())

# print(df.shape)

# print(df['Seller_Type'].unique())
# print(df['Fuel_Type'].unique())
# print(df['Transmission'].unique())

##check missing or null

# print(df.isnull().sum())

# print(df.describe())

# print(df.columns)

final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

# print(final_dataset['Transmission'].unique())
# print(final_dataset['Owner'].unique())
# print(final_dataset['Seller_Type'].unique())

final_dataset['Current_Year']=2021

# print(final_dataset.head())



final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']

# print(final_dataset.head())
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['Current_Year'],axis=1,inplace=True)

##USEING GET DUMMIES FOR feature engineering
final_dataset=pd.get_dummies(final_dataset,drop_first=True)

# print(final_dataset.columns)
# print(final_dataset.head())

# print(final_dataset.corr())
# sns.pairplot(final_dataset)
# sns.show()

corrmat=final_dataset.corr()
top_corr_features=corrmat.index
# plt.figure(figsize=(20,20))
# g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()

x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

# print(x)
# print(y)

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)

# print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(5).plot(kind='barh')
# plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()

##HYPERPARAMERTS

n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]

# print(n_estimators)
max_features=['auto','sqrt']

max_depth=[int(x) for x in linspace(5,30,num=6) ]


min_samples_split=[2,5,10,15,100]

min_samples_leaf=[1,2,5,10]

from sklearn.model_selection import RandomizedSearchCV

random_gird={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf

}

# print(random_gird)
rf_random=RandomizedSearchCV(estimator=rf_reg,param_distributions=random_gird,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)

rf_random.fit(x_train,y_train)

print(rf_random.best_params_)
print(rf_random.best_score_)

# rf_random.save()

predictions=rf_random.predict(x_test)
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)

from sklearn import metrics
print("MAE",metrics.mean_absolute_error(y_test,predictions))
print("MSE",metrics.mean_squared_error(y_test,predictions))
print("RMSR",np.sqrt(metrics.mean_absolute_error(y_test,predictions)))

import pickle

file=open('ramdom_forest_regression.pkl','wb')

pickle.dump(rf_random,file)