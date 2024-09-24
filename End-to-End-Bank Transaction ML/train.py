from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import joblib
import argparse


data = pd.read_csv('./bankdata.csv')

print(data.columns)

X = data.drop('SEX',axis=1)
y = data['SEX']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100,random_state=0)
rf.fit(X_train, y_train)

print('Accuracy: ',accuracy_score(y_test, rf.predict(X_test)))
# print('Confusion Matrix:\n',confusion_matrix(y_test, rf.predict(X_test)))
print('Classification Report:\n',classification_report(y_test,rf.predict(X_test)))
print('ROC AUC Score: ',roc_auc_score(y_test,rf.predict(X_test)))

joblib.dump(rf, "bankdatamodel.pkl")