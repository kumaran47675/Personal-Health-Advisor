import numpy as np
import pandas as pd
import  pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
from sklearn import metrics  
from scikeras.wrappers import KerasClassifier   
from tensorflow.keras.models import Sequential    
from tensorflow.keras.layers import Dense, Activation  
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.ensemble import ExtraTreesRegressor

import time
start = time.time()
df=pd.read_csv("cardio_train.csv",sep=";")
df.drop("id",inplace=True,axis=1)
Y=df["cardio"]
X=df.drop("cardio",axis=1)
#Feature Selection
model=ExtraTreesRegressor()
feat_imp=model.fit(X,Y)
feat_imp.feature_importances_
feat_imp=pd.Series(feat_imp.feature_importances_,index=X.columns)
feat_imp.nlargest(13).plot(kind="barh")
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
seed=10

print("Starting Random forest")
classifier = RandomForestClassifier(verbose=2,random_state=seed)
classifier.fit(X_train, Y_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_random = classifier.predict(X_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("The accuracy is "+str(accuracy_score(Y_test,y_pred_random)))
#Accuracy is 0.710
 
print("Starting Naive Bayes")
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_nb = gnb.predict(X_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("Accuracy is "+str(accuracy_score(Y_test,y_pred_nb)))
#The accuracy is 0.588
 
#Decision tree
print("Starting Decision tree")
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_dt = clf.predict(X_test)
y_pred_dt_roc = clf.predict_proba(X_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("The accuracy is "+str(accuracy_score(y_pred_dt,Y_test)))
#Accuracy is 0.6326285714285714
 
print("Starting Multi layer perceptron")
model = MLPClassifier( max_iter=130, batch_size=1000, alpha=1e-4, activation = 'relu',solver='adam', verbose=10, tol=1e-4, random_state=seed)
model.fit(X_train, Y_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_mlp = model.predict(X_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("Accuracy is "+str(accuracy_score(y_pred_mlp,Y_test)))
#Accuracy is 0.4976
 
print("Starting Gradient boost")
model = GradientBoostingClassifier(n_estimators=20, random_state=seed,verbose=2)
model.fit(X_train, Y_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_gradient = model.predict(X_test)
endtest =time.time()
difftest = endtest-starttest
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Test time: " + str(difftest))
print("Accuracy is"+str(accuracy_score(y_pred_gradient,Y_test)))
#Accuracy is 0.729
 
model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(30, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal'))
model.add(Dense(6,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),callbacks=[monitor],verbose=2,epochs=200,batch_size=1000)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_nn = model.predict(X_test)
y_pred_nn = np.argmax(y_pred_nn,axis=1)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
 
print(model.summary())
print(model.evaluate(X_test,Y_test)[1])
#accuracy is #0.626
diff1=pd.read_csv("CVD_cleaned.csv",sep=",")
diff1["Exercise"]=diff1["Exercise"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Heart_Disease"]=diff1["Heart_Disease"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Skin_Cancer"]=diff1["Skin_Cancer"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Other_Cancer"]=diff1["Other_Cancer"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Depression"]=diff1["Depression"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Diabetes"]=diff1["Diabetes"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Arthritis"]=diff1["Arthritis"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["Sex"]=diff1["Sex"].apply(lambda x: 1 if x=="Male" else 0)
diff1["Smoking_History"]=diff1["Smoking_History"].apply(lambda x: 1 if x=="Yes" else 0)
diff1["General_Health"].unique()
diff1["General_Health"]=diff1["General_Health"].replace("Poor",0)
diff1["General_Health"]=diff1["General_Health"].replace("Fair",1)
diff1["General_Health"]=diff1["General_Health"].replace("Good",2)
diff1["General_Health"]=diff1["General_Health"].replace("Very Good",3)
diff1["General_Health"]=diff1["General_Health"].replace("Excellent",4)
diff1.Checkup.unique()
diff1["Checkup"]=diff1["Checkup"].replace("Never",0)
diff1["Checkup"]=diff1["Checkup"].replace("Within the past year",1)
diff1["Checkup"]=diff1["Checkup"].replace("Within the past 2 years",2)
diff1["Checkup"]=diff1["Checkup"].replace("Within the past 5 years",3)
diff1["Checkup"]=diff1["Checkup"].replace("5 or more years ago",4)
diff1.Age_Category.unique()
category_map = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
                '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
                '70-74': 10, '75-79': 11, '80+': 12}
diff1['Age_Category'] = diff1['Age_Category'].replace(category_map)
sns.heatmap(diff1.corr(),annot=True,cmap="Greens")
diff1.isnull().sum()
X1=diff1.drop("General_Health",axis=1)
Y1=diff1["General_Health"]
model1=ExtraTreesRegressor()
feat_imp1=model1.fit(X1,Y1)
feat_imp1=pd.Series(feat_imp1.feature_importances_,index=X1.columns)
feat_imp1.nlargest(13).plot(kind="barh")
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.25)
 
print("Starting Random forest")
classifier = RandomForestClassifier(verbose=2,random_state=seed)
classifier.fit(X1_train, Y1_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_random = classifier.predict(X1_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("The accuracy is "+str(accuracy_score(Y1_test,y_pred_random)))
#Accuracy is 0.40654544512653146
 
print("Starting Naive Bayes")
gnb = GaussianNB()
gnb.fit(X1_train, Y1_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_nb = gnb.predict(X1_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("Accuracy is "+str(accuracy_score(Y1_test,y_pred_nb)))
#The accuracy is 0.34694485456005386
 
#Decision tree
print("Starting Decision tree")
clf = DecisionTreeClassifier()
clf = clf.fit(X1_train,Y1_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_dt = clf.predict(X1_test)
y_pred_dt_roc = clf.predict_proba(X1_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("The accuracy is "+str(accuracy_score(y_pred_dt,Y1_test)))
#Accuracy is 0.3275442277307224
 
print("Starting Multi layer perceptron")
model = MLPClassifier( max_iter=130, batch_size=1000, alpha=1e-4, activation = 'relu',solver='adam', verbose=10, tol=1e-4, random_state=seed)
model.fit(X1_train, Y1_train)
end = time.time()
diff=end-start
print("Training time: " + str(diff))
starttest = time.time()
y_pred_mlp = model.predict(X1_test)
endtest =time.time()
difftest = endtest-starttest
print("Test time: " + str(difftest))
print("Accuracy is "+str(accuracy_score(y_pred_mlp,Y1_test)))
#Accuracy is 0.4294298961328257