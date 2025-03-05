import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.common import random_state
import seaborn as sns
import streamlit as st

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score


@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv("data/mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

@st.cache_data(persist=True)
def split(df):
    y = df.type
    X = df.drop(columns=['type'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushroom edible or poisonous? 🍄")

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    def plot_metrics(metrics_list):
        
        if 'Confusion Metrics' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
            disp.plot(ax=ax)
            st.pyplot(fig=fig)
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            display.plot(ax)
            st.pyplot(fig)
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            display = PrecisionRecallDisplay(precision=precision, recall=recall)
            display.plot(ax)
            st.pyplot(fig)
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)

    st.sidebar.subheader("Chosse Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine(SVM)", "Logistic Regression", "Random Forest"))
    
    if classifier == "Support Vector Machine(SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regualerization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        
        metrics = st.sidebar.multiselect("Metrics", ('Confusion Metrics', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine(SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", round(accuracy,2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regualerization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        
        metrics = st.sidebar.multiselect("Metrics", ('Confusion Metrics', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", round(accuracy,2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("The number of trees in the forest", 100, 500, step=10, key='n_estimators')
        max_depth = st.sidebar.slider("The maximum depth of the tree", 1, 10, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap')
        metrics = st.sidebar.multiselect("Metrics", ('Confusion Metrics', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, random_state=123)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", round(accuracy,2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)


if __name__=='__main__':
    main()
