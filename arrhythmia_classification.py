import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import requests
from streamlit_lottie import st_lottie
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import altair as alt 
import networkx as nx
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
#from keras.models import load_model
#import tensorflow as tf

# ---- Page Title with Icon ----
st.set_page_config(page_title='Arrhythmia Classification', page_icon=':anatomical_heart:')

# ---- Load GFX Assets ----

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

ecg_gfx = 'https://lottie.host/37bfe504-9620-477e-ad38-f32dfe93f35a/U98tiX7gVS.json'

# ---- Introduction ----

def introduction():
    st_lottie(ecg_gfx, height=250)

    st.title("Introduction")
    introduction_text = """
    - Arrhythmia poses a significant challenge in cardiovascular medicine, marked by irregular heart rhythms with severe health implications like stroke and heart failure.
    - Early arrhythmia detection is pivotal for timely treatment, motivating our model to promptly identify ECG abnormalities for swift healthcare interventions.
    - Our main goal is to develop a machine learning model for accurate differentiation between cardiac arrhythmia presence and absence, initially targeting atrial fibrillation and expanding to cover various arrhythmia types.
    """
    st.write(introduction_text)
    st.write("## Project Methodology")
        # Step 1
    st.subheader("Step 1: Preprocessing and Feature Engineering")
    st.write("This step involves preprocessing the data, engineering features, and downsampling to handle class imbalance.")
    # Step 2
    st.subheader("Step 2: Model selection")
    st.write("In this step, we select baseline models and more sophisticated models such as bagging and boosting. We use techniques like GridsearchCV and Randomized Search for hyperparameter tuning. In addition, we apply Deep Learning models such as Dense Neural Networks and Artificial Neural Networks.")
    # Step 3
    st.subheader("Step 3: Performance evaluation")
    st.write("This step involves systematically evaluating model performance based on metrics like accuracy and recall. A main focus is placed on the number of undetected arrhythmia cases as evident in number of false negatives.")

# ---- Model Loading Function ----
# Define a function to load the models
@st.cache_resource
def load_uci_models():
    uci_models = {
        'Logistic Regression': joblib.load('uci_best_model_LogisticRegression.joblib'),
        'Random Forest': joblib.load('uci_best_model_RandomForestClassifier.joblib'),
        'Support Vector': joblib.load('uci_best_model_SVC.joblib'),
        'Elastic Net': joblib.load('uci_best_model_ElasticNet.joblib'),
        'Gradient Boosting': joblib.load('uci_best_model_GradientBoostingClassifier.joblib'),
        'AdaBoost': joblib.load('uci_best_model_AdaBoostClassifier.joblib'),
        'XGBoost': joblib.load('uci_best_model_XGBClassifier.joblib')
    }
    return uci_models

@st.cache_resource
def load_mit_models():
    mit_models = {
        'Logistic Regression': joblib.load('mit_best_model_LogisticRegression.joblib'),
        'Random Forest': joblib.load('mit_best_model_RandomForestClassifier.joblib'),
#        'Support Vector': joblib.load('mit_best_model_SVC.joblib'),
        'Elastic Net': joblib.load('mit_best_model_ElasticNet.joblib'),
        'Gradient Boosting': joblib.load('mit_best_model_GradientBoostingClassifier.joblib'),
        'AdaBoost': joblib.load('mit_best_model_AdaBoostClassifier.joblib'),
        'XGBoost': joblib.load('mit_best_model_XGBClassifier.joblib')
    }
    return mit_models

@st.cache_resource
def load_deep_learning_models():
    deep_learning_models ={    
        "DNN_relu": joblib.load('mit_best_model_DNN_relu.joblib'),
        "DNN_sigmoid": joblib.load('mit_best_model_DNN_sigmoid.joblib'),
        "DNN_tanh": joblib.load('mit_best_model_DNN_tanh.joblib'),
        "ANN_relu": joblib.load('mit_best_model_ANN_relu.joblib'),
        "ANN_sigmoid": joblib.load('mit_best_model_ANN_sigmoid.joblib'),
        "ANN_tanh": joblib.load('mit_best_model_ANN_tanh.joblib')
    }
    return deep_learning_models

# ---- UCI Bilkent Dataset ----

def uci_bilkent_dataset():
    st.title("UCI-Bilkent Dataset")
    selected_page = st.sidebar.selectbox("Select Page", ["Exploration", "Preprocessing and Feature Engineering", "Modelling"])
    # Read UCI-Bilkent Dataset
    df = pd.read_csv('arrhythmia.csv')
    input_data = pd.read_csv('uci_x_test.csv')
    target_values = pd.read_csv('uci_y_test.csv')

    if selected_page == "Exploration":
        st.write("## Exploratory Data Analysis")
        st.dataframe(df.head(10))
        st.write(df.shape)
        st.dataframe(df.describe())

        if st.checkbox("Show NA"):
            st.dataframe(df.isna().sum())

        # Missing Values 
        df_obj = df.select_dtypes(include=['object']).columns
        for o in df_obj:
            df[o] = pd.to_numeric(df[o], errors='coerce')

        summary_df = df.isnull().sum()
        missing_stats = summary_df[summary_df != 0].reset_index()
        missing_stats.columns = ['column_name', 'no_of_missing']
        missing_stats['percent_missing'] = (missing_stats['no_of_missing'] / len(df)) * 100

        # Plotting the Missing Values
        st.write("## Missing Values")
        fig, ax = plt.subplots()
        ax.bar(missing_stats['column_name'], missing_stats['no_of_missing'])
        ax.set_title('Missing values')
        ax.set_xlabel('Column Names')
        ax.set_ylabel('Missing values')
        ax.set_xticklabels(missing_stats['column_name'], rotation=90)
        for i, value in enumerate(missing_stats['percent_missing']):
            ax.text(i, missing_stats['no_of_missing'][i], f"{value:.1f}%", ha='center', va='bottom')
        ax.set_ylim(0, len(df))
        st.pyplot(fig)

        # Distribution among Patient Age and Sex
        st.write("## Distribution of Patients Age and Sex")
        plt.figure(figsize=(10, 6))
        df.groupby('sex')['age'].plot(kind='hist', alpha=0.5, legend=True)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Distribution of Patients Age by Sex')
        plt.legend(['Female', 'Male'])

        # Pass the figure object to st.pyplot()
        st.pyplot(plt.gcf())

        # Distribution of Classes
        st.write("## Distribution of Classes")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Distribution of Classes')
        st.pyplot(plt.gcf())

    elif selected_page == "Preprocessing and Feature Engineering":
        st.write("## Preprocessing and Feature Engineering")
    
        # Load data
        df = pd.read_csv('UCI-BILKENT_Arrhythmia_Dataset_preprocessed_cleaned_classes_label.csv', sep=',', index_col=0)

        # Separate features and target variable
        X = df.drop(['class','label'], axis=1)  # Features
        y = df['label']  # Target variable

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define a function to perform PCA and return transformed data
        def perform_pca(n_components):
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            return X_pca, pca.explained_variance_ratio_, pca.components_

        # Streamlit app
        st.title("PCA Visualization")
    
        # Slider for selecting number of PCA components
        st.write("Select the Number of PCA Components")
        n_components = st.select_slider("Number of PCA Components", options=[30, 40, 50, 55, 60, 65, 70, 75, 78, 80, 85, 90, 95, 100])

        # Perform PCA
        X_pca, explained_variance_ratio, components = perform_pca(n_components)

        # Display cumulative variance ratio plot
        fig, ax = plt.subplots()
        ax.plot(range(1, n_components+1), explained_variance_ratio.cumsum(), marker='o', linestyle='-')
        ax.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance Ratio")
        ax.set_title("Cumulative Variance Ratio vs. Number of Components")
        st.pyplot(fig)

        # Display explained variance ratio for each component
        st.subheader("Explained Variance Ratio for Each Component")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, n_components+1), explained_variance_ratio, marker='o', linestyle='-')
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Explained Variance Ratio")
        ax2.set_title("Explained Variance Ratio for Each Component")
        st.pyplot(fig2)

        # Display principal components
        st.subheader("Principal Components")
        components_df = pd.DataFrame(components, columns=X.columns)
        st.write(components_df)

    elif selected_page == "Modelling":
#        st.write("## Systematic comparison of different Machine Learning Models for Arrhythmia Classification")
        st.write('### Hyperparameter space for GridSearchCV')
        data = {
            "Model": ["Logistic Regression", "Random Forest", "Support Vector", "Elastic Net", "Gradient Boosting", "AdaBoost", "XGBoost"],
            "Hyperparameter Space": [
                "solver: [liblinear, lbfgs]; C: np.logspace(-4, 2, 9)",
                "n_estimators: [10, 50, 100, 250, 500, 1000]; min_samples_leaf: [1, 3, 5]; max_features: [sqrt, log2]",
                "C: np.logspace(-4, 2, 9); kernel: [linear, rbf]",
                "C: np.logspace(-4, 2, 9); l1_ratio: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]",
                "n_estimators: [50, 100, 200]; learning_rate: [0.01, 0.1, 1.0]; max_depth: [3, 5, 7]",
                "n_estimators: [50, 100, 200]; learning_rate: [0.01, 0.1, 1.0]",
                "n_estimators: [50, 100, 200]; learning_rate: [0.01, 0.1, 1.0]; max_depth: [3, 5, 7]"]
        }
        hyperparameter_table = pd.DataFrame(data)
        st.table(hyperparameter_table)

        # Load multiple models
        models = load_uci_models()

        st.title('Model Results')

        train_data_size = len(df)
        test_data_size = len(input_data)
        st.write("Size of Train Dataset:", train_data_size, '/ Size of Test Dataset:', test_data_size)

        # Model selection widget
        selected_model = st.selectbox('Select Model', list(models.keys()))

        # Define a function to make predictions
        def predict(model, input_data):
            return model.predict(input_data)

        # Make prediction based on selected model
        if selected_model in models:
            prediction = predict(models[selected_model], input_data)

            # Display model attributes
            show_model_attributes = st.checkbox("Show Model Attributes")
            if show_model_attributes:
                st.subheader('Model Attributes:')
                model_attributes_box = st.empty()
                model_attributes = models[selected_model].get_params()
                model_attributes_box.write(model_attributes)

            # Display performance summary
            if hasattr(models[selected_model], 'score'):
                accuracy = models[selected_model].score(input_data, target_values)
                rounded_accuracy = round(accuracy, 4)
                st.subheader('Model Performance Summary:')
                st.write(f'Accuracy: {rounded_accuracy}')

            # Display confusion matrix
            if hasattr(models[selected_model], 'predict'):
                # Display classification report
                st.subheader('Classification Report:')
                report = classification_report(target_values, prediction)
                st.text(report)

                st.subheader('Confusion Matrix:')
                cm = confusion_matrix(target_values, prediction)

                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax)
                ax.set_xlabel('Predicted') 
                ax.set_ylabel('Actual')

                # Display confusion matrix plot
                st.pyplot(fig)
        else:
            st.write('No model selected.')
        

# ---- MIT BIH Dataset ----

def mit_bih_dataset():
    # Read MIT-BIH Dataset
    df = pd.read_csv('MIT-BIH Arrhythmia Database.csv')
    input_data = pd.read_csv('mit_x_test.csv')
    target_values = pd.read_csv('mit_y_test.csv')

    st.title("MIT-BIH Dataset")
    selected_page = st.sidebar.selectbox("Select Page", ["Exploration and Preprocessing", "Modelling", "Deep Learning"])

    # Adjust MIT-BIH Arrhythmia Dataset and Preprocessing needed for both page 1 and 2 
    df_orig = df.copy() 
    class_names = {'N': 'Normal', 'SVEB': 'Supraventricular ectopic beat', 'VEB': 'Ventricular ectopic beat', 'F': 'Fusion beat', 'Q': 'Unknown beat'}
    df['type'] = df['type'].map(class_names)
    df_orig['type'] = df_orig['type'].map(class_names)
    # Create binary target variable
    class_mapping_lambda = lambda x: 0 if x == 'Normal' else 1
    df['label'] = df['type'].apply(class_mapping_lambda)
    df.drop(['type'], axis=1, inplace=True)
    X = df.drop('label', axis=1)
    y = df['label'] 
    majority_class = df[df['label'] == 0]
    minority_class = df[df['label'] == 1]
    downsampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)  
    balanced_df = pd.concat([downsampled_majority, minority_class])
    balanced_df = balanced_df.sample(frac=1, random_state=42)
    X_balanced = balanced_df.drop(['label'], axis=1)
    y_balanced = balanced_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
#    input_data = X_test
#    target_values = y_test

    if selected_page == "Exploration and Preprocessing":
        st.write("## Exploratory Data Analysis and Preprocessing")
        st.write("### Data Exploration")
        st.dataframe(df.head(10))
        st.write(df.shape)
        st.dataframe(df.describe())
        if st.checkbox("Show NA"):
            st.dataframe(df.isna().sum())
        # Plot distribution of types
        plt.figure(figsize=(10, 8))
        ax = sns.countplot(data=df_orig, x="type")
        plt.xticks(rotation=90, fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.xlabel('Type', fontsize=16)
        plt.title('Distribution of Each Type', fontsize=20)
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 9), 
                   textcoords='offset points')
        plt.tight_layout()
        st.pyplot(plt)    
        st.write("### Creation of binary target variable")
        # Plot pie chart   
        label_counts = df['label'].value_counts(normalize=True)
        plt.figure(figsize=(6, 6))
        plt.pie(label_counts, labels=['Normal', 'Arrhythmia'], explode=[0.05, 0.05], autopct="%0.2f%%")
        plt.title('Normal vs. Arrhythmia', fontsize=16)
        #st.pyplot(plt)
        ###
        st.write("### Downsampling to address class imbalance")
        st.write(f"Shape of the original dataset: {df.shape}")
        st.write(f"Shape of the downsampled dataset: {balanced_df.shape}")
        # Pie charts next to each other
        label_counts_original = np.bincount(df['label'].astype(int))
        label_counts_ds = np.bincount(balanced_df['label'].astype(int))
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        axes[0].pie(label_counts_original, labels=['Normal', 'Arrhythmia'], explode=[0.05, 0.05], autopct="%0.2f%%")
        axes[0].set_title('Distribution before downsampling', fontsize=16)
        axes[1].pie(label_counts_ds, labels=['Normal', 'Arrhythmia'], explode=[0.05, 0.05], autopct="%0.2f%%")
        axes[1].set_title('Distribution after downsampling', fontsize=16)
        plt.tight_layout()
        st.pyplot(plt)

    elif selected_page == "Modelling":
#       st.write("## Systematic comparison of different Machine Learning Models for Arrhythmia Classification")
        st.write('### Hyperparameter space for Randomized Search ')
        data = {
             "Model": ["Logistic Regression", "Random Forest", "Support Vector", "Elastic Net", "Gradient Boosting", "AdaBoost", "XGBoost"],
            "Hyperparameter Space": [
        "solver: [liblinear, lbfgs]; C: np.logspace(-4, 2, 9)",
        "n_estimators: [10, 50, 100, 250, 500, 1000]; min_samples_leaf: [1, 3, 5]; max_features: [sqrt, log2]",
        "C: np.logspace(-4, 2, 9); kernel: [linear, rbf]",
        "C: np.logspace(-4, 2, 9); l1_ratio: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]",
        "n_estimators: [50, 100, 200]; learning_rate: [0.01, 0.1, 1.0]; max_depth: [3, 5, 7]",
        "n_estimators: [50, 100, 200]; learning_rate: [0.01, 0.1, 1.0]",
        "n_estimators: [50, 100, 200]; learning_rate: [0.01, 0.1, 1.0]; max_depth: [3, 5, 7]"]
        }
        hyperparameter_table = pd.DataFrame(data)
        st.table(hyperparameter_table)
####
        st.write('### Model Performance Comparison')
        models = ["Logistic Regression", "Random Forest", "Elastic Net", "Gradient Boosting", "Ada Boosting", "XG Boosting"]
        best_parameters = ["C: 3.162; penalty: l2", "max_features: log2; min_samples_leaf: 1; n_estimators: 250",
                       "C: 0.01; l1_ratio: 0.4; max_iter: 1000; penalty: elasticnet; solver: saga",
                       "learning_rate: 0.1; max_depth: 7; n_estimators: 200",
                       "learning_rate: 1.0; n_estimators: 200",
                       "learning_rate: 1.0; max_depth: 7; n_estimators: 200"]
        train_accuracy = [0.88, 1.00, 0.86, 1.00, 0.96, 1.00]
        test_accuracy = [0.88, 0.97, 0.86, 0.98, 0.95, 0.98]
        precision = [0.90, 0.98, 0.91, 0.98, 0.95, 0.98]
        recall = [0.87, 0.98, 0.80, 0.98, 0.94, 0.98]
        f1_score = [0.88, 0.98, 0.85, 0.98, 0.95, 0.98]
        auroc_score = [0.88, 0.98, 0.86, 0.98, 0.95, 0.98]

        # Barplot with selectbox 
        bar_width = 0.15
        index = np.arange(len(models))
        selected_model = st.select_slider("Select Model", options=list(models.keys()))  # Get the keys of the dictionary
        fig, ax = plt.subplots(figsize=(12, 8))
        model_index = list(models.keys()).index(selected_model)  # Find the index of the selected model key
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        #colors = ['lightblue', 'lightgreen', 'lightcoral', 'yellow']
        for i, model in enumerate(models):
            alpha = 1 if i == model_index else 0.4
            ax.bar(index[i] - 2*bar_width, test_accuracy[i], bar_width, color=colors[0], edgecolor='black', hatch='/', alpha=alpha)
            ax.bar(index[i] - bar_width, train_accuracy[i], bar_width, color=colors[1], edgecolor='black', hatch='\\', alpha=alpha)
            ax.bar(index[i], recall[i], bar_width, color=colors[2], edgecolor='black', hatch='x', alpha=alpha)
            ax.bar(index[i] + bar_width, recall[i], bar_width, color=colors[3], edgecolor='black', hatch='.', alpha=alpha)
        ax.set_xlabel('Model')
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of Model Performances')
        ax.set_xticks(index)    
        ax.set_xticklabels(list(models.keys()))  # Use the keys of the dictionary
        ax.legend(['Test Accuracy', 'Train Accuracy', 'Test Recall', 'Train Recall'], bbox_to_anchor=(1, 1), loc='upper left')
        st.pyplot(fig)

        ####
        
        st.write("### Comparison of Confusion Matrices")
        image_path = "Figure_18.png"  
        image = open(image_path, 'rb').read()
        st.image(image, caption='Overall, Gradient Boost shows the smallest number of false negative. ', use_column_width=True)  
        
        #####
        st.write("### Model Performance")

        # Load multiple models
        models = load_mit_models()

        st.title('Model Results')

        train_data_size = len(df)
        test_data_size = len(input_data)
        st.write("Size of Train Dataset:", train_data_size, '/ Size of Test Dataset:', test_data_size)

        # Model selection widget
        selected_model = st.selectbox('Select Model', list(models.keys()))

        # Define a function to make predictions
        def predict(model, input_data):
            return model.predict(input_data)

        # Make prediction based on selected model
        if selected_model in models:
            prediction = predict(models[selected_model], input_data)

            # Display model attributes
            show_model_attributes = st.checkbox("Show Model Attributes")
            if show_model_attributes:
                st.subheader('Model Attributes:')
                model_attributes_box = st.empty()
                model_attributes = models[selected_model].get_params()
                model_attributes_box.write(model_attributes)

            # Display performance summary
            if hasattr(models[selected_model], 'score'):
                accuracy = models[selected_model].score(input_data, target_values)
                rounded_accuracy = round(accuracy, 4)
                st.subheader('Model Performance Summary:')
                st.write(f'Accuracy: {rounded_accuracy}')

            # Display confusion matrix
            if hasattr(models[selected_model], 'predict'):
                # Display classification report
                st.subheader('Classification Report:')
                report = classification_report(target_values, prediction)
                st.text(report)

                st.subheader('Confusion Matrix:')
                cm = confusion_matrix(target_values, prediction)

                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax)
                ax.set_xlabel('Predicted') 
                ax.set_ylabel('Actual')

                # Display confusion matrix plot
                st.pyplot(fig)
        else:
            st.write('No model selected.')
        
    elif selected_page == "Deep Learning":
        st.write("## Comparison of different Neural Network architectures for Arrhythmia Classification")
        st.write('### Dense Neural Networks')
        st.write("#### Confusion Matrices for DNNs with different activation functions")
        image_path = "Figure_19.png"  
        image = open(image_path, 'rb').read()
        st.image(image, caption='', use_column_width=True)
        DNN_text = """
        DNN with input dimensions (None, 32), three hidden dense layers of 10, 8, 6, and 3 neurons, (493 trainable parameters total), trained over 500 epochs with an adaptive Adam optimizer
        """
        st.write(DNN_text)
        
        st.write('### Artificial Neural Networks')
        st.write("#### Confusion Matrices for ANNs with different activation functions")
        image_path = "Figure_20.png"  
        image = open(image_path, 'rb').read()
        st.image(image, caption='', use_column_width=True)
        ANN_text = """
        ANN with three hidden dense layers four neurons each, with a sigmoid activation function in the output layer for binary classification, trained over 500 epochs with adaptive Adam optimizer
        """
        st.write(ANN_text)

        st.write("### Precision-Recall Curves for DNN and ANN Trials")
        image_path = "Figure_22.png"  
        image = open(image_path, 'rb').read()
        st.image(image, caption='', use_column_width=True)

# ---- Conclusions ----

def summary():
    st.header("Summary")
    summary_text = """
    - Gradient Boosting achieved best recall with a high accuracy of 98%
    - DNN and ANN achieved accuracies ranging from 95% to 96%
    - Deep Learning models got outperformed by simpler ensemble models
    - One possible explanation for this discrepancy could be the dataset's size and complexity
    """

# Visualizations
    data = {
        'Model': ['Gradient Boosting', 'Random Forest', 'XG Boosting', 'DNN', 'Ada Boosting', 'Logistic Regression', 'ANN', 'Elastic Net'],
        'Test Accuracy': [0.98, 0.97, 0.98, 0.95, 0.95, 0.88, 0.94, 0.86],
        'Recall': [0.98, 0.98, 0.98, 0.95, 0.94, 0.87, 0.90, 0.80],
        'False negatives': [44, 47, 48, 49, 121, 347, 111, 419]
    }
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='False negatives')
    # Add 1 to the index to start from 1
    df_sorted.index = range(1, len(df_sorted) + 1)
    # Add interactivity to the bar chart using Altair
    bars = alt.Chart(df_sorted).mark_bar().encode(
        y=alt.Y('Model:N', title='Model', sort='x'),  # Sorting by descending order of false negatives
        x=alt.X('False negatives:Q', title='Number of False Negatives'),
        color=alt.condition(
            alt.datum.Model == df_sorted.iloc[0]['Model'],
            alt.value('orange'),  # Color for the model with the fewest false negatives
            alt.value('steelblue')  # Default color for other models
        ),
        tooltip=['Test Accuracy:Q', 'Recall:Q']  # Show accuracy and recall on hover
    ).properties(
        title=''
    ).configure_axis(

    ).configure_title(
        fontSize=16,
        color='black'  
    ).configure_legend(
        titleColor='black',  
        labelColor='black'   
    )

    # Display the Altair chart
    st.write("### Model Ratings in terms of False Negatives:")
    st.altair_chart(bars, use_container_width=True)

    # Display conclusion text as bullet points with larger font size
    st.write(summary_text)

def explorations():
    st.header("Future Explorations")
    explorations_text = """
    - Further refinement through hyperparameter tuning
    - Usage of advanced deep learning methodologies such as encoding-decoding techniques
    - Exploring different sampling strategies like including patient data from other hospitals or using GAN's 
    - Consultation with cardiologists and machine learning engineers for further improvement
    - Classification of more nuanced heartbeat conditions
    - Optimizing pre-processing for better applicability in real life data
    """
    st.write(explorations_text)
    
def relevance():
    st.header("Relevance")
    relevance_text = """
    - Model studies demonstrate significant potential for deployment in clinical settings, such as hospitals and healthcare facilities
    - Robust performance and ability to minimize false negatives of models like Gradient Boosting, XGBoost, and Random Forest, makes them valuable tools in arrhythmia diagnosis
    - Interpretable models enhance transparency and understanding, facilitating collaboration between machine learning experts and healthcare professionals
    - End user devices like smartwatches or portable ECG trackers could provide preventive care
    - Scalability and individualization possibilities can improve patient care over populations, applications or time
    """
    st.write(relevance_text)
    
def conclusions():
    st.title("Conclusions")
    selected_page = st.sidebar.selectbox("Select Page", ["Summary", "Future Explorations", "Relevance"])
    if selected_page == "Summary":
        summary()
    elif selected_page == "Future Explorations":
        explorations()
    elif selected_page == "Relevance":
        relevance()
        

# ---- Main / Sidebar ----

def main():
    st.sidebar.title("Classification of Arrhythmia")
    page = st.sidebar.radio("Contents", ["Introduction", "UCI-Bilkent Dataset", "MIT-BIH Dataset", "Conclusions"])
    if page == "Introduction":
        introduction()
    elif page == "UCI-Bilkent Dataset":
        uci_bilkent_dataset()
    elif page == "MIT-BIH Dataset":
        mit_bih_dataset()
    elif page == "Conclusions":
        conclusions()

if __name__ == "__main__":
    main()
