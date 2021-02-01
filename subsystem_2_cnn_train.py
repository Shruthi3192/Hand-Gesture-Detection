import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten

# load dataset
df = pd.read_csv("Dataset_Subsystem_2.csv", header=None)
IP=df.loc[:,4:]
IP.columns = IP.iloc[0]
IP = IP[1:]
IP=IP.apply(pd.to_numeric)

OP=df.loc[:, 2:3]
OP=pd.DataFrame(OP[2].str.cat(OP[3],sep="_"))
OP = pd.DataFrame(OP.rename(columns={2: "labels"}))
OP = OP.iloc[1:]
#display(OP)
#print(OP.shape[0])
num_samples = OP.shape[0]
#print(OP.iloc[2].labels)
label=np.ones((num_samples,),dtype = int)

#print(num_samples)
for i in range(num_samples):
   # print(i)
#    print(OP.iloc[i+1].labels)
    if(OP.iloc[i].labels == "open_palm"):
        label[i] = 0
    if(OP.iloc[i].labels == "open_dorsal"):
        label[i] = 1
    if(OP.iloc[i].labels == "fist_palm"):
        label[i] = 2
    if(OP.iloc[i].labels == "fist_dorsal"):
        label[i] = 3
    if(OP.iloc[i].labels == "three_fingers_palm"):
        label[i] = 4
    if(OP.iloc[i].labels == "three_fingers_dorsal"):
        label[i] = 5
    
encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
dummy_y = np_utils.to_categorical(encoded_Y)
#display(dummy_y)

 
X_train, X_test, y_train, y_test = train_test_split(IP
                                                    , dummy_y
                                                    , test_size=0.20
                                                    , shuffle=True
                                                    , random_state=32
                                                   )

print(X_train.shape)
print(y_train.shape)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


X_train = X_train.reshape(-1,80,1)
X_test = X_test.reshape(-1,80,1)
print(X_train.shape)
print(X_test.shape)


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(80,1)))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size=10, verbose=2)
# evaluate model
_, accuracy = model.evaluate(X_test, y_test, batch_size=10, verbose=2)
print(accuracy)

model_json = model.to_json()
with open("model_cnn.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_cnn.h5")
print("Saved model to disk")