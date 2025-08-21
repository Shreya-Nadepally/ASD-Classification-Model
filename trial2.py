import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv("dataset.csv")  # Replace with your actual dataset path

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df.drop(columns=['Class/ASD'])  # Replace 'Class/ASD' with your actual target column name
y = df['Class/ASD']

# Normalize features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using Chi-square
k = 13  # Number of top features to select
chi2_selector = SelectKBest(score_func=chi2, k=k)
X_kbest = chi2_selector.fit_transform(X_scaled, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

