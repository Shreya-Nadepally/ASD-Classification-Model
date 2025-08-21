import streamlit as st
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# ----- Cuckoo Search Implementation -----
class SimpleCuckooSearch:
    def __init__(self, objective_function, n=15, pa=0.25, alpha=0.01, lower_bound=None, upper_bound=None, dimension=10, max_iter=50):
        self.obj_func = objective_function
        self.n = n
        self.pa = pa
        self.alpha = alpha
        self.lower_bound = lower_bound or [0] * dimension
        self.upper_bound = upper_bound or [1] * dimension
        self.dimension = dimension
        self.max_iter = max_iter
        self.nests = [self.random_solution() for _ in range(n)]
        self.fitness = [self.obj_func(nest) for nest in self.nests]
        self.best_nest = self.nests[np.argmax(self.fitness)]
        self.best_fitness = max(self.fitness)

    def random_solution(self):
        return [random.uniform(self.lower_bound[i], self.upper_bound[i]) for i in range(self.dimension)]

    def get_best_solution(self):
        return self.best_nest

    def update(self):
        for _ in range(self.max_iter):
            for i in range(self.n):
                new_solution = [x + self.alpha * random.gauss(0, 1) for x in self.nests[i]]
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_fitness = self.obj_func(new_solution)
                if new_fitness > self.fitness[i]:
                    self.nests[i] = new_solution
                    self.fitness[i] = new_fitness
                    if new_fitness > self.best_fitness:
                        self.best_nest = new_solution
                        self.best_fitness = new_fitness
            for i in range(self.n):
                if random.random() < self.pa:
                    self.nests[i] = self.random_solution()
                    self.fitness[i] = self.obj_func(self.nests[i])

# ----- Streamlit App -----
st.set_page_config(page_title="ASD Classifier", layout="wide")

def main():
    st.title("🧠 ASD Classification with Cuckoo Search Optimization")
    st.markdown("Upload a dataset, select a model, and optimize classification performance using **Cuckoo Search Algorithm**.")

    with st.sidebar:
        st.header("📂 Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        st.markdown("---")
        model_name = st.selectbox("🧪 Choose Classifier", [
            "SVM", "KNN", "Decision Tree", "Logistic Regression", "Random Forest",
            "XGBoost", "AdaBoost", "Gradient Boosting"
        ])
        run_button = st.button("🚀 Run Optimization")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Preprocessing
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        X = df.drop(columns=['Class/ASD'])
        y = df['Class/ASD']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model Constructor
        def get_model(name, params):
            if name == "SVM":
                return SVC(C=params[0], gamma=params[1])
            elif name == "KNN":
                return KNeighborsClassifier(n_neighbors=int(params[0]))
            elif name == "Decision Tree":
                return DecisionTreeClassifier(max_depth=int(params[0]))
            elif name == "Logistic Regression":
                return LogisticRegression(C=params[0], max_iter=500)
            elif name == "Random Forest":
                return RandomForestClassifier(n_estimators=int(params[0]), max_depth=int(params[1]))
            elif name == "XGBoost":
                return XGBClassifier(n_estimators=int(params[0]), max_depth=int(params[1]), learning_rate=params[2],
                                     use_label_encoder=False, eval_metric='logloss')
            elif name == "AdaBoost":
                return AdaBoostClassifier(n_estimators=int(params[0]), learning_rate=params[1])
            elif name == "Gradient Boosting":
                return GradientBoostingClassifier(n_estimators=int(params[0]), max_depth=int(params[1]), learning_rate=params[2])

        # Objective Function
        def objective(params):
            model = get_model(model_name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        # Define hyperparameter search space
        if run_button:
            st.subheader("🔍 Running Cuckoo Search Optimization...")

            if model_name == "KNN":
                bounds = [[1, 20]]
            elif model_name == "Decision Tree":
                bounds = [[1, 20]]
            elif model_name == "Random Forest":
                bounds = [[10, 200], [1, 20]]
            elif model_name == "SVM":
                bounds = [[0.01, 10], [0.001, 1]]
            elif model_name == "Logistic Regression":
                bounds = [[0.01, 10]]
            elif model_name == "XGBoost":
                bounds = [[10, 200], [1, 10], [0.01, 0.5]]
            elif model_name == "AdaBoost":
                bounds = [[10, 200], [0.01, 1]]
            elif model_name == "Gradient Boosting":
                bounds = [[10, 200], [1, 10], [0.01, 0.5]]

            lower_bound = [b[0] for b in bounds]
            upper_bound = [b[1] for b in bounds]
            dimension = len(bounds)

            cs = SimpleCuckooSearch(objective, lower_bound=lower_bound, upper_bound=upper_bound, dimension=dimension, max_iter=30)
            cs.update()

            best_params = cs.get_best_solution()
            best_accuracy = cs.best_fitness

            st.success(f"🎯 Best Accuracy: {best_accuracy:.2%}")
            st.code(f"Best Parameters: {np.round(best_params, 3)}")
            st.metric("Accuracy", f"{best_accuracy:.2%}")
            st.balloons()

# Entry point
if __name__ == "__main__":
    main()




