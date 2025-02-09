# Password Similarity Model

## Overview

This project implements a password similarity model using machine learning techniques to analyze and cluster breached passwords. The model helps in assessing the similarity between a given password and a dataset of compromised passwords, providing insights into password strength and security.

## Dataset used
(Breached passwords dataset)[http://kaggle.com/datasets/emanuellopez/breached-passwords]

## Features

- **Data Loading**: Loads a dataset of breached passwords.
- **Data Preprocessing**: Masks passwords to create a feature representation.
- **Feature Engineering**: Generates a feature matrix for clustering.
- **Clustering**: Uses Agglomerative Clustering to group similar passwords.
- **Similarity Calculation**: Computes the similarity percentage between a user-provided password and the clusters of breached passwords.
- **Model Persistence**: Saves the trained model and scaler for future use.

## Requirements

To run this project, you need to have the following Python packages installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- joblib

You can install the required packages using pip:
```bash
pip install -r requirements.txt
```


## Usage

1. **Load the Dataset**: Ensure you have the `breachcompilation.csv` file in the `data` directory.
2. **Run the Jupyter Notebook**: Open the `password_similarity_model.ipynb` file in Jupyter Notebook or JupyterLab.
3. **Execute the Cells**: Run each cell in the notebook sequentially to load the data, preprocess it, train the model, and compute password similarity.

### Example

To compute the similarity of a password, you can use the following code snippet:
```python
test_password = "your_password_here"
similarity, cluster = compute_password_similarity(
test_password,
clustering,
X_scaled,
scaler
)
print(f"Password similarity: {similarity:.2f}%")
print(f"Most similar cluster: {cluster}")
```


## Model Saving and Loading

The trained model and scaler can be saved to a file using the following code:
```python

## Model Saving and Loading

The trained model and scaler can be saved to a file using the following code:
```


To load the model later, use:
```python
loaded_model = joblib.load('model3.pkl')
clustering = loaded_model['clustering']
scaler = loaded_model['scaler']
```
