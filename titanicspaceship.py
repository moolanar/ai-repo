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
