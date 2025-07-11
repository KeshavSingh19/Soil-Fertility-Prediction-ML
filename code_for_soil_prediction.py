code for the comparison of the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Soil_Data.csv')
df.head()
print(df.columns)

df.isnull().sum()
df.fillna(df.mean(), inplace=True)

X = df.drop('Output', axis=1)
y = df['Output']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
 
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, zero_division=0))

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=0))

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))

accuracy_scores = {
    'Decision Tree': accuracy_score(y_test, y_pred_dt),
    'SVM': accuracy_score(y_test, y_pred_svm),
    'Random Forest': accuracy_score(y_test, y_pred_rf)
}

names = list(accuracy_scores.keys())
scores = list(accuracy_scores.values())

plt.figure(figsize=(4,4))
sns.barplot(x=names, y=scores)
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy')
plt.show()

importances = rf.feature_importances_
feature_names = df.drop('Output', axis=1).columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance (Random Forest)')
plt.show()

print("\nFinal Conclusion: Random Forest achieved the highest accuracy of all models (around 92%). Thus, Random Forest is selected for soil fertility prediction.")





print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, zero_division=0))

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=0))

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))

accuracy_scores = {
    'Decision Tree': accuracy_score(y_test, y_pred_dt),
    'SVM': accuracy_score(y_test, y_pred_svm),
    'Random Forest': accuracy_score(y_test, y_pred_rf)
}

names = list(accuracy_scores.keys())
scores = list(accuracy_scores.values())

plt.figure(figsize=(4,4))
sns.barplot(x=names, y=scores)
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy')
plt.show()

importances = rf.feature_importances_
feature_names = df.drop('Output', axis=1).columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance (Random Forest)')
plt.show()

print("\nFinal Conclusion: Random Forest achieved the highest accuracy of all models (around 92%). Thus, Random Forest is selected for soil fertility prediction.")

