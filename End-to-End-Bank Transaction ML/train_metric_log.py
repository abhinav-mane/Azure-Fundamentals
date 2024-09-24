from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import joblib
import argparse
from azureml.core import Run
import os

run = Run.get_context()

data = pd.read_csv('./bankdata.csv')

print(data.columns)

X = data.drop('SEX',axis=1)
y = data['SEX']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100,random_state=0)
rf.fit(X_train, y_train)

run.log("Parameter",rf.get_params())

print('Accuracy: ',accuracy_score(y_test, rf.predict(X_test)))
run.log("Accuracy",accuracy_score(y_test, rf.predict(X_test)))
# print('Confusion Matrix:\n',confusion_matrix(y_test, rf.predict(X_test)))
print('Classification Report:\n',classification_report(y_test,rf.predict(X_test)))
print('ROC AUC Score: ',roc_auc_score(y_test,rf.predict(X_test)))

# Create the outputs directory (note the 's' at the end)
os.makedirs('outputs', exist_ok=True)

# Save your model to the outputs directory
joblib.dump(rf, 'outputs/bankdatamodel.pkl') # note this outputs folder has a special convention in azure ML; you need to use outputs folder always to save your model
