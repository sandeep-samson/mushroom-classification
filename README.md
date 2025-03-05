# Mushroom Classification Web App Documentation

## Overview

The Mushroom Classification Web App is a Streamlit-based application designed to classify mushrooms as either edible or poisonous using machine learning models. The application provides an interactive interface for users to select different classification models and visualize performance metrics.

## Features

- **Data Loading**: The app loads a dataset of mushroom descriptions and preprocesses it for classification.
- **Model Selection**: Users can choose between three classification models: Support Vector Machine (SVM), Logistic Regression, and Random Forest.
- **Hyperparameter Tuning**: Each model comes with adjustable hyperparameters to optimize performance.
- **Performance Metrics**: The app displays accuracy, precision, recall, confusion matrix, ROC curve, and precision-recall curve for model evaluation.
- **Data Visualization**: Users can visualize the raw dataset and various performance metrics through interactive plots.

## Installation and Setup

### Prerequisites

- Docker
- Docker Compose

### Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t mushroom-classification-app .
   ```

3. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 mushroom-classification-app
   ```

4. **Access the App**:
   Open a web browser and navigate to `http://localhost:8501`.

## Application Structure

### `app.py`

This is the main application file that defines the Streamlit interface and the core functionality of the app. It includes:

- **Data Loading and Preprocessing**: The dataset is loaded and preprocessed using `pandas` and `LabelEncoder`.
- **Model Training and Evaluation**: Users can train and evaluate different models using the sidebar controls.
- **Metrics Visualization**: The app visualizes various performance metrics using `matplotlib` and `seaborn`.

### `Dockerfile`

The Dockerfile defines the environment for the application. It sets up a Python environment, installs dependencies, and configures the Streamlit app to run inside a Docker container.

### Dataset

The dataset includes descriptions of hypothetical mushroom samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each sample is labeled as edible or poisonous based on various features such as cap shape, cap color, odor, gill attachment, etc.

## Usage

1. **Select a Classifier**: Choose from SVM, Logistic Regression, or Random Forest.
2. **Adjust Hyperparameters**: Use the sidebar controls to adjust model hyperparameters.
3. **Train and Evaluate**: Click the "Classify" button to train the model and view performance metrics.
4. **Visualize Metrics**: Select metrics to visualize, such as confusion matrix, ROC curve, and precision-recall curve.

## Conclusion

The Mushroom Classification Web App provides an interactive and user-friendly interface for classifying mushrooms using machine learning. With adjustable hyperparameters and visual performance metrics, it serves as a valuable tool for both educational and practical purposes.
