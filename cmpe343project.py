
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

dataset = pd.read_csv('diabetes.csv')


from pandas_profiling import ProfileReport
file = ProfileReport(dataset)
file.to_file(output_file='output.html')


# Check the shape of the data
# the first 8 columns are features while the last one
# is the supervised label (1 = has diabetes, 0 = no diabetes)
dataset.shape
# (768,9)

# Visualise a table with the first rows of the dataset, to
# better understand the data format
dataset.head()


#Data correlation matrix
corr = dataset.corr()
corr


#matplotlib inline
import seaborn as sns
sns.heatmap(corr, annot = True)


# Calculate the median value for BMI
median_bmi = dataset['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
dataset['BMI'] = dataset['BMI'].replace(to_replace=0, value=median_bmi)

# Calculate the median value for BloodP
median_bloodp = dataset['BloodPressure'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
dataset['BloodPressure'] = dataset['BloodPressure'].replace(to_replace=0, value=median_bloodp)

# Calculate the median value for PlGlcConc
median_plglcconc = dataset['Glucose'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
dataset['Glucose'] = dataset['Glucose'].replace(to_replace=0, value=median_plglcconc)

# Calculate the median value for SkinThick
median_skinthick = dataset['SkinThickness'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
dataset['SkinThickness'] = dataset['SkinThickness'].replace(to_replace=0, value=median_skinthick)

# Calculate the median value for TwoHourSerIns
median_twohourserins = dataset['Insulin'].median()
# Substitute it in the TwoHourSerIns column of the
# dataset where values are 0
dataset['Insulin'] = dataset['Insulin'].replace(to_replace=0, value=median_twohourserins)



#Splitting the data into dependent and independent variables
Y = dataset.Outcome
x = dataset.drop('Outcome', axis = 1)
columns = x.columns

#feature scalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)
data_x = pd.DataFrame(X, columns = columns)


#Splitting the data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, Y, test_size = 0.15, random_state = 45)


from sklearn.metrics import accuracy_score


# ------------Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
classifier_RFC.fit(x_train, y_train)
y_pred_RFC = classifier_RFC.predict(x_test)
print('------------Random Forest Classifier')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_RFC)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_RFC, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_RFC, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_RFC, average="macro")))



# ------------Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0)
classifier_LR.fit(x_train, y_train)
y_pred_LR = classifier_LR.predict(x_test)
print('------------Logistic Regression')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_LR)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_LR, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_LR, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_LR, average="macro")))




# ------------K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(x_train, y_train)
y_pred_KNN = classifier_KNN.predict(x_test)
print('------------K-Nearest Neighbors (K-NN)')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_KNN)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_KNN, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_KNN, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_KNN, average="macro")))


# ------------Support Vector Machine (SVM)
from sklearn.svm import SVC
classifier_SVC = SVC(kernel = 'linear', random_state = 0)
classifier_SVC.fit(x_train, y_train)
y_pred_SVC = classifier_SVC.predict(x_test)
print('------------Support Vector Machine (SVM)')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_SVC)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_SVC, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_SVC, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_SVC, average="macro")))




# ------------Kernel SVM
from sklearn.svm import SVC
classifier_SVC_rbf = SVC(kernel = 'rbf', random_state = 0)
classifier_SVC_rbf.fit(x_train, y_train)
y_pred_SVC_rbf = classifier_SVC_rbf.predict(x_test)
print('------------Kernel SVM')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_SVC_rbf)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_SVC_rbf, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_SVC_rbf, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_SVC_rbf, average="macro")))



# ------------Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(x_train, y_train)
y_pred_NB = classifier_NB.predict(x_test)
print('------------Naive Bayes')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_NB)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_NB, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_NB, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_NB, average="macro")))



# ------------Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifier_DTC = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DTC.fit(x_train, y_train)
y_pred_DTC = classifier_DTC.predict(x_test)
print('------------Decision Tree Classification')
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_DTC)))
print('f1 score: {:.2f}'.format(f1_score(y_test, y_pred_DTC, average="macro")))
print('precision score: {:.2f}'.format(precision_score(y_test, y_pred_DTC, average="macro")))
print('recall score: {:.2f}'.format(recall_score(y_test, y_pred_DTC, average="macro")))


'''

 ##### Experimental results #####

------------Random Forest Classifier
accuracy: 0.77
f1 score: 0.75
precision score: 0.75
recall score: 0.75
------------Logistic Regression
accuracy: 0.73
f1 score: 0.70
precision score: 0.71
recall score: 0.70
------------K-Nearest Neighbors (K-NN)
accuracy: 0.69
f1 score: 0.66
precision score: 0.66
recall score: 0.66
------------Support Vector Machine (SVM)
accuracy: 0.73
f1 score: 0.70
precision score: 0.71
recall score: 0.70
------------Kernel SVM
accuracy: 0.74
f1 score: 0.71
precision score: 0.72
recall score: 0.70
------------Naive Bayes
accuracy: 0.72
f1 score: 0.69
precision score: 0.69
recall score: 0.69
------------Decision Tree Classification
accuracy: 0.64
f1 score: 0.60
precision score: 0.60
recall score: 0.60
'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_RFC)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(cm, cmap=plt.cm.Greys, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
