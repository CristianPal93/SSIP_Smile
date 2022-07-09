#importing the libraries
import os
import cv2
from sklearn.model_selection import train_test_split
def create_set(path,classes):
    data={}
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    for class_type in classes:
        data_dir = path+class_type
        for img in os.listdir(data_dir):
            pic = cv2.imread(os.path.join(data_dir,img))
            pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
            pic = pic.flatten()
            print(pic.shape)
            data['data'].append(pic)
            data['label'].append(class_type)
            data['filename'].append(img)
    return data

path = './dataset/'
classes = ['0','1']
data = create_set(path,classes)
X = data['data']
y = data['label']
print(X[0].shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, shuffle=True,random_state=42,)

from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=10)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))