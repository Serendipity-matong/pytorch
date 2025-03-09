import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载
train = pd.read_csv(r'D:\KaggleNotebook\train.csv',index_col=0)
test = pd.read_csv(r'D:\KaggleNotebook\test.csv',index_col=0)
train_null = train.isna().sum()
test_null  = test.isna().sum()
train_null.drop(labels='SalePrice', axis=0, inplace=True)

X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0) #training, validation, and test set
y = train[['SalePrice']] #target for

plt.figure(figsize=(25,8))
plt.title('Number of missing rows')
missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()
missing_count.columns = ['features','sum']
sns.barplot(x='features',y='sum', data = missing_count)
plt.show()

X.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)

categorical_feature = X.select_dtypes(include='object').columns.tolist()

numerical_feature   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_feature   = [col for col in numerical_feature if col != 'Id']

discrete_feature    = [col for col in numerical_feature if len(X[col].unique()) < 50]
continuous_feature  = [col for col in numerical_feature if col not in discrete_feature]

print(f'Number of Categorical Feature : {len(categorical_feature)}')
print(f'Number of Numerical Feature   : {len(numerical_feature)}')
print(f'Number of Discrete Feature    : {len(discrete_feature)}')
print(f'Number of Continous Feature   : {len(continuous_feature)}')

fig , axes = plt.subplots(nrows=4, ncols=5, figsize=(20,15))

for i, feature in enumerate(continuous_feature):
    sns.histplot(data= train, x=feature, ax=axes[i%4,i//4], color='red')
    sns.histplot(data= test,  x=feature, ax=axes[i%4,i//4],  color='darkblue')
plt.show()

fig = plt.figure(figsize=(30,30))
for index,col in enumerate(train[numerical_feature]):
    plt.subplot(6,6,index+1)
    sns.histplot(train[numerical_feature].loc[:,col].dropna(), kde=False)
    if index+1 > len(numerical_feature):
        pass
fig.tight_layout(pad=1.0)

plt.show()

num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_num.append(i)

overfit_num = list(overfit_num)
X = X.drop(overfit_num, axis=1)

print("Numerical Features with >96% of the same value: ",overfit_num)

cat = X.select_dtypes(include=['object']).copy()
fig = plt.figure(figsize=(18,20))
for index in range(len(cat.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat.iloc[:,index], data=cat.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
plt.show()

cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_cat.append(i)

overfit_cat = list(overfit_cat)
X = X.drop(overfit_cat, axis=1)
print("Categorical Features with >96% of the same value: ",overfit_cat)

numeric = X.select_dtypes(exclude=['object']).copy()
plt.figure(figsize=(14,12))
correlation = numeric.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='coolwarm',annot=True, fmt='.2g')
plt.show()

X.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)

numeric = X.select_dtypes(exclude=['object']).copy()
plt.figure(figsize=(14,12))
correlation = numeric.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='coolwarm', annot=True, fmt='.2g')

numeric_versus_sales = train.select_dtypes(exclude=['object']).copy()
fig = plt.figure(figsize=(20,20))
for index in range(len(numeric_versus_sales.columns)):
    plt.subplot(10,5,index+1)
    sns.scatterplot(x=numeric_versus_sales.iloc[:,index], y='SalePrice', data=numeric_versus_sales.dropna())
fig.tight_layout(pad=1.0)
plt.show()

fig = plt.figure(figsize=(14,15))
for index,col in enumerate(numeric):
    plt.subplot(7,7,index+1)
    sns.boxplot(y=col, data=numeric.dropna())
    if index+1 > len(numeric):
        pass
fig.tight_layout(pad=1.0)
plt.show()

out_col = ['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea','GarageArea']
fig = plt.figure(figsize=(10,8))
for index,col in enumerate(out_col):
    plt.subplot(1,6,index+1)
    sns.boxplot(y=col, data=train)
fig.tight_layout(pad=1.5)
plt.show()

#Drop outliers based on thresholds
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index)
train = train.drop(train[(train['GarageArea'] > 1200) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['TotalBsmtSF'] > 4000) & (train['SalePrice'] < 200000)].index)
train = train.drop(train[train['LotFrontage'] > 200].index)
train = train.drop(train[train['LotArea'] > 50000].index)
train = train.drop(train[train['BsmtFinSF1'] > 3000].index)

out_col = ['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea','GarageArea']
fig = plt.figure(figsize=(10,8))
for index,col in enumerate(out_col):
    plt.subplot(1,6,index+1)
    sns.boxplot(y=col, data=train)
fig.tight_layout(pad=1.5)
plt.show()

print(pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(15))

#categorical to be filled with NA only
cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1',
       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu',"KitchenQual",
       "HeatingQC",'ExterQual','ExterCond']

X[cat] = X[cat].fillna("NA")
X[cat].isnull().sum()

#categorical filled with mode
cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else X[col].mode()[0]))
X[cols].isnull().sum()

print("Mean of LotFrontage: ", X['LotFrontage'].mean())
print("Mean of GarageArea: ", X['GarageArea'].mean())

neigh_lot = X.groupby('Neighborhood')['LotFrontage'].mean().reset_index(name='LotFrontage_mean')
neigh_garage = X.groupby('Neighborhood')['GarageArea'].mean().reset_index(name='GarageArea_mean')

fig, axes = plt.subplots(1,2,figsize=(22,8))
axes[0].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='LotFrontage_mean', data=neigh_lot, ax=axes[0])
axes[1].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='GarageArea_mean', data=neigh_garage, ax=axes[1])

#Impute with neighboor means for wide distributed features dependent on locality.
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))

#Impute missing values in MSZoning and transform MSSubClass to string
X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
X['MSSubClass'] = X['MSSubClass'].apply(str)

#numerical
cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]
X[cont] = X[cont] = X[cont].fillna(X[cont].mean())
X[cont].isnull().sum()

ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

ord_col = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond',
           'FireplaceQu']
for col in ord_col:
    X[col] = X[col].map(ordinal_map)

fin_col = ['BsmtFinType1', 'BsmtFinType2']
for col in fin_col:
    X[col] = X[col].map(fintype_map)

X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)

# X['Fence'] = X['Fence'].map(fence_map)

X['TotalLot'] = X['LotFrontage'] + X['LotArea']
X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']
X['TotalBath'] = X['FullBath'] + X['HalfBath']
X['Total_Close_Live_Area'] = X['GrLivArea'] + X['TotalBsmtSF']
X['Outside_live_area'] =  X['WoodDeckSF'] + X['OpenPorchSF'] + X['EnclosedPorch']+ X['ScreenPorch']
X['Total_usable_area'] = X['Total_Close_Live_Area'] + X['Outside_live_area']
X['Area_Quality_Indicator'] = X['Total_usable_area'] * X['OverallQual']
X['Area_Qual_Cond_Indicator'] = X['Total_usable_area'] * X['OverallQual']* X['OverallCond']

column = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF']

for col in column:
    col_name = col+'_bin'
    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)

X = pd.get_dummies(X)

plt.figure(figsize=(10,6))
plt.title("Before transformation of SalePrice")
dist = sns.displot(train['SalePrice'], kde=True, stat='density')

plt.figure(figsize=(10,6))
plt.title("After transformation of SalePrice")
dist = sns.displot(train['SalePrice'], kde=True, stat='density')

y["SalePrice"] = np.log(y['SalePrice'])
x = X.loc[train.index]
y = y.loc[train.index]
X_test = X.loc[test.index]

cols = x.select_dtypes(np.number).columns
transformer = RobustScaler().fit(x[cols])
x[cols] = transformer.transform(x[cols])
X_test[cols] = transformer.transform(X_test[cols])

pd.DataFrame(x.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(5)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Shape of training data (num_rows, num_columns)
print(X_test.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_test.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

from xgboost import XGBRegressor
from sklearn import ensemble
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold

useless_feature = {}
useful_feature = {}

ridge = Ridge(random_state=12)

param_lst = {
 'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
  'max_iter': [50, 100, 200, 300, 400, 500, 1000]
}

ridge_reg = RandomizedSearchCV(estimator = ridge, param_distributions = param_lst,
                             n_iter = 100, scoring = 'neg_root_mean_squared_error',
                             cv = 5)

ridge_search = ridge_reg.fit(X_train, y_train)

# XGB with tune hyperparameters
best_param = ridge_search.best_params_
best_param = {'max_iter': 50, 'alpha': 10.0}
ridge = Ridge(**best_param)

ridge.fit(X_train, y_train)
preds = ridge.predict(X_valid)
preds_test_huber = ridge.predict(X_test)
mae_ridge = mean_absolute_error(y_valid, preds)


coefficients = ridge.coef_

features_importance = pd.DataFrame({
    'feature': ridge.feature_names_in_,    # get feature
    'coefficient': coefficients.flatten()            # get coefficient
})

features_importance['coefficient'] = abs(features_importance['coefficient'])
features_importance = features_importance.sort_values(by='coefficient', ascending=False).reset_index(drop=True)


top_10 = features_importance.loc[:10,:]   # get top ten
plt.figure(figsize=(6,8))
sns.barplot(x= top_10['coefficient'], y= top_10['feature'], color='blue')
plt.title('Feature Importance Ridge Regression')

# get useless features
zero = features_importance[features_importance['coefficient'] < 0.001 ].sort_values(by='feature', ascending=True)

# put into dictionary
for feature in zero['feature']:
    useless_feature[feature] = useless_feature.get(feature,0) + 1

# threshold for useful features
threshold = 0.02

# capture useful features
high_coef = features_importance[features_importance['coefficient'] >= threshold].sort_values(by='coefficient', ascending=False)
high_coef = high_coef.reset_index(drop=True)

for i, feature in enumerate(high_coef['feature']):
    #access based on weight
    useful_feature[feature] = useful_feature.get(feature, 0) + high_coef['coefficient'][i]

plt.figure(figsize=(10, 8))
sns.barplot(x=top_10['coefficient'], y=top_10['feature'], palette='viridis')
plt.title('Top 10 Feature Importance - Ridge Regression', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.show()

print(preds_test_huber)
preds_test_huber = preds_test_huber.flatten()


preds_test_huber = np.exp(preds_test_huber)
output = pd.DataFrame({'Id': X_test.index,
                      'SalePrice': preds_test_huber})
output.to_csv(r'D:\KaggleNotebook\submission.csv', index=False)


