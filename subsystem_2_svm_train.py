import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import svm
from sklearn import metrics

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

num_samples = OP.shape[0]
label=np.ones((num_samples,),dtype = int)

for i in range(num_samples):
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

print(label)        
#encoder = LabelEncoder()
#encoder.fit(label)
#encoded_Y = encoder.transform(label)
#dummy_y = np_utils.to_categorical(encoded_Y)

#print(dummy_y)
#Import svm model

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(IP, label)

#Predict the response for test dataset
#y_pred = clf.predict(IP)
#print(y_pred)

#Import scikit-learn metrics module for accuracy calculation

sample = pd.read_csv("landmarks.csv", header=None)

#Predict the response for test dataset
y_pred = clf.predict(sample)
print(y_pred)

#sample = pd.read_csv("landmarks_open_palm.csv", header=None)
#sample = sample.iloc[:,:40]

# Model Accuracy: how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(label, y_pred))

