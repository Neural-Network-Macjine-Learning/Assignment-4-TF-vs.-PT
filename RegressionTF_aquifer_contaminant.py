import time
# get the start time
st = time.process_time()

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


df = pd.read_csv(
    'E:/Machine Learning/Assign4/Data_as_4/Data_as4.csv')


#Split dataset into train and test
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

#Creating label and feature for train dataframe
y_train=train_dataset['Fluoride_avg']
feature_names = ['WellDepth','Elevation','Rainfall','Tmin','claytotal_l', 'awc_l','ph_r'] # INPUT DATA FEATURES
X_train=train_dataset[feature_names]

#Convert to tensors
y_train=tf.convert_to_tensor(y_train)
X_train=tf.convert_to_tensor(X_train)


#Next make a regression model predict the Fluoride content. 
normalize = layers.Normalization() #Normalize data
normalize.adapt(X_train)


fluoride_model = tf.keras.Sequential([
  normalize,
  layers.Dense(32, activation='relu'),
  layers.Dense(1)
])

#Use mse as the loss function and Adam as optimizer
fluoride_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))

#To train that model, pass the features and labels to Model.fit:

fluoride_model.fit(X_train, y_train, epochs=10)

#Creating label and feature for test dataframe
X_test=test_dataset[feature_names]
X_test=tf.convert_to_tensor(X_test)
normalize.adapt(X_test)
y_test=test_dataset['Fluoride_avg']
#Convert to tensors
y_test=tf.convert_to_tensor(y_test)


print("Result: ", fluoride_model.evaluate(X_test,  y_test, verbose=2))

#Model summary and weights

fluoride_model.summary()
print("Weights in zeroth layer:", fluoride_model.layers[0].weights)

print("Weights in layer 1:", fluoride_model.layers[1].weights)

print("Weights in layer 2:",fluoride_model.layers[2].weights)



fluoride_model.save("Ogallala_fluoride_model")
# get the end time
et = time.process_time()
elapsed_time = (et - st)*1000
print('CPU Execution time:', elapsed_time, 'milliseconds')

