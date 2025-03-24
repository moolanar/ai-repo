import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/ml/annclassification/Churn_Modelling.csv")


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)


data = data.drop(['CustomerId' , 'Surname'], axis=1)
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

ohe = OneHotEncoder()
geo_ohe = ohe.fit_transform(data[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_ohe, columns=ohe.get_feature_names_out(['Geography']))

fulldata = pd.concat([data, geo_df], axis=1)



y = fulldata['Exited']
x = fulldata.drop(['Geography', 'Exited', 'EstimatedSalary'], axis=1)

ixtr, xt , iytr,  yt = train_test_split(x, y, test_size=0.1)
xtr,  xv ,  ytr,  yv = train_test_split(ixtr, iytr, test_size=0.25)

print(ixtr.shape, iytr.shape)
print(xtr.shape, ytr.shape)
print(xv.shape, yv.shape)
print(xt.shape, yt.shape)

scaler = StandardScaler()
xtr = scaler.fit_transform(xtr)
xt = scaler.transform(xt)

print(xtr)

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(12,), activation='relu'),
    keras.layers.Dropout(rate = 0.1),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

#model.compile(optimizer='adam' , loss ='mean_absolute_error' , metrics=['mae']) # regression
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #classification

model.summary()

hist = model.fit(xtr, ytr, validation_data=(xv, yv), epochs=100)

y_pred = model.predict(xt)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yt, y_pred)

print(cm)
