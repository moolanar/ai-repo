import numpy as np
from scipy.stats import randint, uniform
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt  
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder,StandardScaler , MinMaxScaler 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error, make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split , GridSearchCV, RandomizedSearchCV
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import loguniform, uniform
from xgboost import XGBRegressor, XGBRFRegressor
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 300)
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from datetime import datetime

# Record the start time
start_time = datetime.now() 
logy=True

df_final=loaddata() 

#print(dtr.loc[dtr['KitchenAbvGr'] < 1 ,['KitchenAbvGr','KitchenQual']]) 

numeric_feats = []
numeric_feats = df_final.dtypes[df_final.dtypes != "object"].index 
skewed_feats = df_final[numeric_feats].apply(lambda x: x.skew())
skewed_features = skewed_feats[skewed_feats > 0.75].index
for feat in skewed_features:
    if feat != 'SalePrice':
        df_final[feat] = np.log1p(df_final[feat])

print(df_final['SalePrice'])
# Record the start time
start_time = datetime.now()  

qualitative  = [f for f in df_final.columns if df_final.dtypes[f] == 'object']
print(qualitative)

for c in qualitative:
    df_final[c]=df_final[c].astype(str) 

cat_features = ['Utilities','BsmtFinType1','BsmtFinType2',
'Functional','BsmtExposure','Foundation','GarageFinish',
'CentralAir','PavedDrive','Street','LandSlope','LotShape','Heating','BldgType','MSZoning',
'HeatingQC','ExterQual','ExterCond','BsmtQual','BsmtCond','KitchenQual',
'GarageQual','GarageCond','Neighborhood','Exterior1st','Exterior2nd']

(X_train, X_test , y_train, y_test) = traintestsplit(df_final, 0.3) 

df_t = pd.read_csv("C:\ml\houcingprdiction\hous_test.csv")
dtest = df_final[1460:]
dtest.drop('SalePrice', axis=1, inplace=True)

#num_feats = X_train.dtypes[X_train.dtypes != "object"].index 
#stdscalar = StandardScaler()
#X_train[num_feats] = stdscalar.fit_transform(X_train[num_feats])
#dtest[num_feats] = stdscalar.fit(X_train[num_feats])

model = CatBoostRegressor(loss_function='RMSE' , learning_rate= 0.01, l2_leaf_reg= 1, depth= 6,
        cat_features=qualitative,random_seed=42, n_estimators=3500, 
        verbose=False)

model.fit(X_train, y_train)
final_pred = model.predict(dtest)

print(final_pred, np.exp(final_pred))

submission = pd.DataFrame({
    'Id': df_t['Id'],
    'SalePrice': np.exp(final_pred)
})
submission.to_csv('my_submission.csv', index=False)    

def loaddata():
    #=================================================================
    df_tr = pd.read_csv("C:\ml\houcingprdiction\hous_train.csv")
    df_t = pd.read_csv("C:\ml\houcingprdiction\hous_test.csv")
    df_t
    df_t['SalePrice']=0
    df=pd.concat([df_tr, df_t], axis=0) 
    #=================================================================

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    for c in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'] :
        df[c] = np.where(df['TotalBsmtSF'] < 1, 'NA' , df[c] )

    for c in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'] :
        df[c] = np.where(df['TotalBsmtSF'] < 1, 0 , df[c] )

    for c in ['MasVnrType'] :
        df[c] = np.where(df['MasVnrArea'] < 1, 'None' , df[c] )

    for c in ['KitchenQual'] :
        df[c] = np.where(df['KitchenAbvGr'] < 1, 'PO' , df[c] )


    for c in ['PoolQC'] :
        df[c] = np.where(df['PoolArea'] < 1, 'NA' , df[c] )

    for c in ['MiscFeature'] :
        df[c] = np.where(df['MiscVal'] < 1, 'NA' , df[c] )

    for c in ['FireplaceQu'] :
        df[c] = np.where(df['Fireplaces'] < 1, 'NA' , df[c] )

    #'GarageYrBlt','GarageCars'
    for c in ['GarageType','GarageFinish','GarageQual','GarageCond']:
        df[c] = np.where(df['GarageArea'] < 1, 'NA' , df[c] )

    for col in ('GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        df[col] = df[col].fillna(0)

    cols_mean = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageArea']
    cols_median = ['GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath','GarageCars']

    for c in cols_mean:
        df[c] = df[c].fillna(df[c].mean()) 

    for c in cols_median:
        df[c] = df[c].fillna(df[c].median())  

    drop_null_cols = df.isna().sum()[df.isna().sum()>1000].index
    print(np.array(drop_null_cols))
    df_final = df.drop(columns=np.array(drop_null_cols))
    print(df_final.shape)
    return df_final

def traintestsplit(df_final , param):

    dtrain = df_final[:1460]
    dtest = df_final[1460:]

    X = dtrain.copy()    
    Y = X['SalePrice']
    X.drop('SalePrice',axis=1,inplace=True)
    dtrain.shape, X.shape, Y.shape  
    
    Y1=Y
    if logy:
        Y1=np.log(Y) 

    xtr, xt, y_train, y_test = train_test_split(X, Y1, test_size=param)
 
    X_train = xtr.copy()
    X_test = xt.copy()
    print(X_train.shape, X_test.shape )
    #X_train[['BsmtQual_encoded','BsmtCond_encoded','KitchenQual_encoded']]   
    return  X_train, X_test , y_train, y_test

def printscores():
    #model.fit(X_train, y_train)
    Y_pred = model.predict(X_test) 
    if logy:
        Y1_pred = np.exp(Y_pred)
        y1_test = np.exp(y_test)
    print(mean_absolute_error(y1_test, Y1_pred) , root_mean_squared_error(y_test, Y_pred), model.best_score_)  
    #sns.scatterplot(x=y_test, y= Y_pred)

def hyperparamtune():

  (X_train, X_test , y_train, y_test) = traintestsplit(df_final, 0.3)
  xtr,vxtr,ytr,vytr = train_test_split(X_train, y_train,test_size=0.05)
  
  scoring = {
      'rmse': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
      'mae': 'neg_mean_absolute_error',
      'r2': 'r2'
  }
  
  catbst = CatBoostRegressor(loss_function='RMSE' , 
              learning_rate=0.05 , 
              n_estimators=3500, 
              #one_hot_max_size=10,
              cat_features=qualitative,random_seed=42, 
              verbose=False)
  
  param_grid = {
      'depth': [1,2,3,4,5,6,7,8,9],    
      'learning_rate': [0.01, 0.05, 0.1],
      'l2_leaf_reg': [9,8,7,6,5,4,3,2,1] 
  }
  
  param_grid1 = {
      'depth': [6, 8, 10],
      'learning_rate': [0.01, 0.05, 0.1],
      'l2_leaf_reg': [1, 3, 5]
  }
  
  random_search = RandomizedSearchCV(
      estimator=catbst,    param_distributions=param_grid,scoring=scoring,
      n_iter=50, cv=5, refit='rmse', random_state=42, verbose=True, n_jobs=-1
  )
  
  random_search.fit(X_train, y_train)
  
  print("Best parameters found: ", random_search.best_params_)
  print("Best score (negative mean squared error): ", random_search.best_score_)
  best_model = random_search.best_estimator_
  y_pred = best_model.predict(X_test)
  final_mse = mean_squared_error(y_test, y_pred)
  print(f"Final Mean Squared Error on test set: {final_mse}")  
