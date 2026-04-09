import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the dataset
# Ensure the file 'processed_wind_lstm_ready(1).xlsx - Sheet1.csv' is in your working directory
file_path = 'processed_wind_lstm_ready(1).xlsx - Sheet1.csv'
df = pd.read_csv(file_path)

# 2. Prepare Features and Target
# The dataset has 14 input columns and 1 target column
X = df.drop(columns=['target_wind_speed']).values
y = df['target_wind_speed'].values

# 3. Reshape for LSTM
# LSTM requires input shape: (samples, time_steps, features)
# Here, we have 14 features which represent 7 time steps with 2 variables each.
time_steps = 7
n_features = 2
X_reshaped = X.reshape((X.shape[0], time_steps, n_features))

# 4. Split into Training and Testing sets
# We use shuffle=False for time-series data to maintain chronological order
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, shuffle=False)

# 5. Build the LSTM Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(time_steps, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the Model
print("Starting model training...")
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1
)

# 7. Predictions and Visualization
y_pred = model.predict(X_test)

plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual Wind Speed (Normalized)', color='steelblue')
plt.plot(y_pred, label='LSTM Predicted Wind Speed', color='orangered', linestyle='--')
plt.title('Multivariate Wind Speed Prediction using LSTM')
plt.xlabel('Time Steps (Test Set)')
plt.ylabel('Normalized Wind Speed')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 8. Model Evaluation
mse = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Mean Squared Error on Test Data: {mse:.5f}")