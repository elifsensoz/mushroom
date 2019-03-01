import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
import seaborn as sns

data = pd.read_csv("mushrooms.csv")
#print(data.isnull().sum())
print(data.shape)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only

scaler = StandardScaler()
X=scaler.fit_transform(X)

plt.figure(figsize=(14,12))
sns.heatmap(data.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
plt.savefig("outputs/correlation.png", bbox_inches='tight')

pca = PCA()
pca.fit_transform(X)
covariance=pca.get_covariance()
explained_variance=pca.explained_variance_


features_list = data.columns
features_list = features_list.drop('class')
print(features_list)
x_pos = np.arange(len(features_list))
plt.figure(figsize=(6, 4))
plt.bar(range(22), explained_variance, alpha=0.5, align='center',label='individual explained variance')
plt.xticks(x_pos, features_list, rotation='vertical')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.tight_layout()
plt.savefig("outputs/explained_variance.png", bbox_inches='tight')


N=data.values
print(N.shape)

pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
plt.savefig("outputs/pca.png",bbox_inches='tight')

kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0 : 'b',
                   1 : 'r'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.savefig("outputs/cluster.png",bbox_inches='tight')

pca_modified=PCA(n_components=17)
pca_modified.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)


t1 = time.time()
model_naive = GaussianNB()
model_naive.fit(X_train, y_train)
t2 = time.time()
g_learning_time = t2-t1
print("Gaussian learning time: "+ str(g_learning_time))

y_prob = model_naive.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
t3 = time.time()
G_prediction_time = t3 - t2
print("Gaussian prediction time: "+ str(G_prediction_time))

model_naive.score(X_test, y_pred)
scores = (cross_val_score(model_naive, X, y, cv=5, scoring='accuracy')).mean()

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
auc_roc=metrics.classification_report(y_test,y_pred)
accuracy1 = metrics.accuracy_score(y_test,y_pred)
print ("Gaussian accuracy: ", + accuracy1)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("outputs/ROC_Gaussian.png",bbox_inches='tight')

t1 = time.time()
model_RR=RandomForestClassifier()
model_RR.fit(X_train,y_train)
t2 = time.time()
r_learning_time = t2 - t1
print("Random Forest learning time: "+ str(r_learning_time))

y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
t3 = time.time()
R_prediction_time = t3 - t2
print("Random Forest prediction time: "+ str(R_prediction_time))

model_RR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
accuracy2 = metrics.accuracy_score(y_test,y_pred)
print ("Random forest accuracy: ", + accuracy2)

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("outputs/ROC_RandomForest.png",bbox_inches='tight')

objects = ('G Learning Time', 'G Prediction Time', 'G Accuracy', 'R Learning Time', 'R Prediction Time', 'R Accuracy',)
y_pos = np.arange(len(objects))
performance = [g_learning_time,G_prediction_time,accuracy1,g_learning_time,R_prediction_time,accuracy2]

plt.figure(figsize=(10,10)) 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects,rotation='vertical')
plt.savefig("outputs/results.png",bbox_inches='tight')
