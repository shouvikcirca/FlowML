import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import mlflow


mlflow.sklearn.autolog()
# Read data
dataset_path = 'data/income_evaluation.csv'
df = pd.read_csv(dataset_path)

# Separate features and target
y = df[' income']
X = df.drop([' income'], axis=1)

# Data Cleaning
categorical = [' workclass',' education',' marital-status',' occupation',' relationship',' race',' sex',' native-country', ' income']
continuous = [col for col in df.columns if col not in categorical ]
categorical.remove(' income')
X[' workclass'].fillna(X[' workclass'].mode()[0], inplace=True)
X[' occupation'].fillna(X[' occupation'].mode()[0], inplace=True)
X[' native-country'].fillna(X[' native-country'].mode()[0], inplace=True)

# Data Preprocessing
# encode categorical variables with one-hot encoding
ohe = OneHotEncoder()
ohe.fit(X[categorical])
enc_df = pd.DataFrame(ohe.transform(X[categorical]).toarray(), columns=ohe.get_feature_names())

X = X[continuous].join(enc_df)

# Creating train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 569)

# Train model
rfc = RandomForestClassifier(random_state=1)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print(X_test)
X_test['prediction']=y_pred

print('-----')
print(X_test)





# Evaluating trained model
#accuracy = accuracy_score(y_test, y_pred) *100
#print('{}%'.format(accuracy))



