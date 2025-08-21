import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import csv
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_clean_dataset(file_path):
    try:
        # Load dataset with proper quoting
        df = pd.read_csv(file_path, 
                        on_bad_lines='skip',
                        quoting=csv.QUOTE_MINIMAL,
                        escapechar='\\')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.strip('"')
        print("✅ Dataset loaded successfully!")
        print("Columns found:", df.columns.tolist())
        
        # Check for target column (try common variations)
        target_col = None
        possible_targets = ['ASD', 'Autism', 'Diagnosis', 'Class/ASD', 'Label']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            raise ValueError(f"Target column not found. Available columns: {df.columns.tolist()}")
        
        print(f"Using target column: '{target_col}'")
        
        # Clean text data
        text_cols = ['Text', 'Class', 'Sign']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.strip('"')
        
        # Convert target to numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
        
        return df, target_col
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

# Main execution
if __name__ == "__main__":
    file_path = "Dataset.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    # Load and clean data
    df, target_col = load_and_clean_dataset(file_path)
    print("\nSample data:")
    print(df.head(3).to_string())
    
    # Data Preparation
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models
    models = {
        "SVM": SVC(),
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    
    # Parameter grids for tuning
    param_grids = {
        "SVM": {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1, 1]},
        "K-NN": {"n_neighbors": [3, 5, 7, 9]},
        "Decision Tree": {"max_depth": [3, 5, 7, 10]},
        "Logistic Regression": {"C": [0.1, 1, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]},
        "XGBoost": {"learning_rate": [0.01, 0.1], "n_estimators": [50, 100]},
        "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.1, 1]},
        "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
    }
    
    # Evaluate models
    results = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n📊 Model Evaluation Results:")
    for name, model in models.items():
        print(f"\n⚙️ Training {name}...")
        try:
            grid_search = GridSearchCV(model, param_grids[name], cv=kf, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            
            results.append({
                'Model': name,
                'Accuracy (%)': f"{acc:.2f}",
                'Best Parameters': grid_search.best_params_
            })
            
            print(f"{name} Accuracy: {acc:.2f}%")
            print(f"Best params: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\n🏆 Final Results:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('model_results.csv', index=False)
    print("\n💾 Results saved to 'model_results.csv'")
    
    # Show best model
    if not results_df.empty:
        best_model = results_df.iloc[results_df['Accuracy (%)'].astype(float).idxmax()]
        print(f"\n🎯 Best Model: {best_model['Model']} with {best_model['Accuracy (%)']}% accuracy")
        print(f"Best Parameters: {best_model['Best Parameters']}")