import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load the datasets
yes = pd.read_csv('yes.csv')
yoo = pd.read_csv('ref.csv')

# Remove spaces from column names in yoo
yoo.columns = [col.strip() for col in yoo.columns]

# Combine the datasets horizontally
data = pd.concat([yoo, yes], axis=1)

# Define feature and target columns
feature_columns = ['A(410)', 'B(435)', 'C(460)', 'D(485)', 'E(510)', 'F(535)', 'G(560)', 'H(585)', 'R(610)',
                   'I(645)', 'S(680)', 'J(705)', 'T(730)', 'U(760)', 'V(810)', 'W(860)', 'K(900)', 'L(940)']
target_columns = ['pH', 'EC (dS/m)', 'OC (%)', 'P (kg/ha)', 'K (kg/ha)', 'Ca (meq/100g)', 'Mg (meq/100g)', 'S (ppm)',
                  'Fe (ppm)', 'Mn (ppm)', 'Cu (ppm)', 'Zn (ppm)', 'B (ppm)']

# Fill missing values in feature and target columns with the median
imputer = SimpleImputer(strategy='median')
data[feature_columns] = imputer.fit_transform(data[feature_columns])
data[target_columns] = imputer.fit_transform(data[target_columns])

# Apply Savitzky-Golay smoothing to the feature columns
data[feature_columns] = savgol_filter(data[feature_columns], window_length=5, polyorder=2, axis=0)

# Split the data into features and target variables
X = data[feature_columns]
y = data[target_columns]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find the optimal number of components
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = []
for n in range(1, min(X_train.shape[0], X_train.shape[1]) + 1):
    plsr = PLSRegression(n_components=n)
    mse = -cross_val_score(plsr, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    mse_scores.append(mse)

optimal_components = np.argmin(mse_scores) + 1

# PLSR Model with optimal number of components
plsr = PLSRegression(n_components=optimal_components)
plsr.fit(X_train, y_train)

# Function to predict with PLSR
def predict_with_plsr(model, features):
    features = np.array(features).reshape(1, -1)
    predictions = model.predict(features)
    return predictions[0]

# Streamlit app layout
st.title('PLSR Model Prediction Interface')

st.header('Input Features')
feature_values = []
for feature in feature_columns:
    value = st.number_input(f'Enter value for {feature}', value=0.0)
    feature_values.append(value)

if st.button('Predict'):
    predictions = predict_with_plsr(plsr, feature_values)
    st.header('Predicted Values')
    for i, target in enumerate(target_columns):
        st.write(f"{target}: {predictions[i]}")

    # Evaluate and display model performance on test set
    y_pred_test = plsr.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    st.write(f"Model Performance on Test Set:")
    st.write(f"R²: {r2_test}")
    st.write(f"RMSE: {rmse_test}")

    # Plot the results for each target variable
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    axes = axes.ravel()
    for i, target in enumerate(target_columns):
        axes[i].scatter(y_test[target], y_pred_test[:, i], color='red', label='Predicted')
        axes[i].plot([y_test[target].min(), y_test[target].max()], [y_test[target].min(), y_test[target].max()], 'k--', lw=2)
        axes[i].set_xlabel('Measured')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{target}')
        axes[i].legend()
    st.pyplot(fig)


