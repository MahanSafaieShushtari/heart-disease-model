# from scikeras.wrappers import kerasClassifier
import pandas as pd 
from tensorflow import keras as ks 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

df=pd.read_csv("heart.csv")
labels = df["HeartDisease"]
dataset= df.drop(["HeartDisease"], axis=1)
labelen=LabelEncoder()
dataset= dataset.apply(labelen.fit_transform)
xtrain,xtest,ytrain,ytest= train_test_split(dataset,labels,test_size=0.2)

xtrain=np.asarray(xtrain)
xtest=np.asarray(xtest)
ytrain,ytest=np.asarray(ytrain),np.asarray(ytest)
normalizer=MinMaxScaler()
xtrain=normalizer.fit_transform(xtrain,(0,1))
xtest=normalizer.fit_transform(xtest,(0,1))

model=ks.Sequential()
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(units=178,activation="relu"))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(units=1,activation="sigmoid"))
model.compile(loss=tf.losses.binary_crossentropy,optimizer=tf.optimizers.Adam(),metrics=["accuracy"])
model.build(input_shape=(None,11))
model.summary()
hist=model.fit(xtrain,ytrain,epochs=100,validation_data=(xtest,ytest),batch_size=28,
               callbacks=ks.callbacks.EarlyStopping(monitor="val_loss",min_delta=0.05,patience=40,restore_best_weights=True))


plt.plot(hist.history['val_accuracy'],color="red")
plt.plot(hist.history['accuracy'],color="green")

plt.show()

