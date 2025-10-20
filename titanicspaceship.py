import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt 
from catboost  import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 
pd.set_option('future.no_silent_downcasting', True)

#===============================================

cat_feats = ['HomePlanet' ,'CryoSleep'  ,'Destination' ,'VIP' , 'serviceflag', 'deck' ,'side' , 'isSingle' ]
model = CatBoostClassifier( cat_features=cat_feats, verbose=False ,
                           learning_rate=0.05 , 
            depth=4, 
            #l2_leaf_reg=10, 
            n_estimators=2500
            )
model.fit(xtr, ytr,
        eval_set=(xe, ye),
        early_stopping_rounds=30,  # Stops if validation metric doesn't improve for 50 rounds
        use_best_model=True 
)
pred = model.predict(xt)
accuracy_score(yt, pred)

#============================================


tX = ts_p.drop(columns=['Transported'])
tP = model.predict(tX)

submission = pd.DataFrame({
    'PassengerId': ts['PassengerId'],
    'Transported': tP
})

submission['Transported'] = submission['Transported'].replace({0:'False',  1:'True'})

submission.to_csv('titanic_spaceship_submission.csv', index=False)  

#====================================================

def preprocess(df):
    plt.figure(figsize=(20, 6))
 
    df = df.reset_index()
    df['HomePlanet'] = np.where( (df['HomePlanet'].isna() ) & (df['Cabin'].str.startswith('G')) , 'Earth' , df['HomePlanet']) 
    df['HomePlanet'] = np.where( (df['HomePlanet'].isna() ) & (df['Cabin'].str.startswith('A')) , 'Europa' , df['HomePlanet']) 
    df['HomePlanet'] = np.where( (df['HomePlanet'].isna() ) & (df['Cabin'].str.startswith('B')) , 'Europa' , df['HomePlanet']) 
    df['HomePlanet'] = np.where( (df['HomePlanet'].isna() ) & (df['Cabin'].str.startswith('C')) , 'Europa' , df['HomePlanet'])    


    df['RoomService']=df['RoomService'].fillna(0)
    df['FoodCourt']=df['FoodCourt'].fillna(0)
    df['ShoppingMall']=df['ShoppingMall'].fillna(0)
    df['Spa']=df['Spa'].fillna(0)
    df['VRDeck']=df['VRDeck'].fillna(0)

    df['TotalService'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    df['CryoSleep'] = df['CryoSleep'].ffill()
    df['VIP'] = df['VIP'].ffill()

    df = df.sort_values(by= ['HomePlanet','Destination', 'PassengerId'])
    df['Cabin'] = np.where(df['PassengerId'].str.endswith('01'), df['Cabin'].bfill(), df['Cabin'].ffill())
    df['Destination'] = np.where(df['PassengerId'].str.endswith('01'), df['Destination'].bfill(), df['Destination'].ffill())
    df['Destination'] = df['Destination'].ffill()
    df = df.sort_index()

    #df['HomePlanet'] = df['HomePlanet'].ffill()
    

    df['CryoSleep'] = df['CryoSleep'].replace({0:False, 1:True})
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP'] = df['VIP'].replace({0:False, 1:True})
    df['VIP'] = df['VIP'].astype(int)
    df['Transported'] = df['Transported'].replace({0:False, 1:True})
    df['Transported'] = df['Transported'].astype(int)

    df['serviceflag'] = np.where(df['TotalService'] > 100, 1 , 0)

    cacols = ['deck','num', 'side' ]
    df[cacols] = df['Cabin'].str.split('/', expand=True)
    df[['group', 'pnum']] = df['PassengerId'].str.split('_', expand=True)
    df['pnum'] = df['pnum'].astype(int) 
    df['group'] = df['group'].astype(int) 

    dt = preprocesshometown(df, 'F', ['Mars', 'Earth']) 
    dt = preprocesshometown(df, 'D', ['Mars', 'Europa']) 
    dt = preprocesshometown(df, 'E', ['Mars', 'Europa', 'Earth']) 
    dt['HomePlanet'] = np.where( dt['HomePlanet'].isna() &  (dt['Destination'] == '55 Cancri e') , 'Europa' , dt['HomePlanet'])  
    dt['HomePlanet'] = np.where( dt['HomePlanet'].isna() &  (dt['Destination'] != '55 Cancri e') , 'Earth' , dt['HomePlanet'])      


    #df['HomePlanet1'] = df.groupby('deck')['HomePlanet'].transform(lambda x: (x.median()))
    #df['Destination1'] = df.groupby('deck')['Destination'].transform(lambda x: (x.median()))

    df['isSingle'] = df.groupby('group')['group'].transform('count')==1
    df['Age1medn'] = df.groupby('deck')['Age'].transform(lambda x: (x.median()))
    df['Age2medn'] = df.groupby('group')['Age'].transform(lambda x: (x.median()))
    df['Age'] = np.where(df['Age'].isna , df['Age1medn'] , df['Age']) 

    df = df.drop(columns=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'
                        ,'Name', 'Cabin' , 'Age1medn' , 'Age2medn', 'PassengerId', 'mask'
                        ] )
    
    return df

#====================================

def preprocesshometown(df , deck , home):
    d = df.copy() 

    d1=d[( d['deck']==deck ) ]
    dup =d1[['group' , 'HomePlanet']]  
    nodup = dup.drop_duplicates() 
    #home = ['Mars', 'Earth']
    for h in home: 
        df['mask'] = False
        enodup = nodup[nodup['HomePlanet']==h]
        my_set = set(enodup['group']) 
        df['mask'] = df['group'].isin(my_set)
        df['HomePlanet'] = np.where(df['mask'] & df['HomePlanet'].isna() , h , df['HomePlanet'])
    return df

full_df = preprocess(df_all) 
tr_p = full_df[:8693]
ts_p = full_df[8693:]
tr.shape, tr_p.shape 

X = tr_p.drop(columns=['Transported'])
Y = tr_p['Transported']

xtr, xt1, ytr, yt1 = train_test_split(X, Y, test_size=0.03)
xe, xt, ye, yt = train_test_split(xt1, yt1, test_size=0.5)

xtr.shape, xt.shape, xe.shape, xt.shape, ye.shape, yt.shape
