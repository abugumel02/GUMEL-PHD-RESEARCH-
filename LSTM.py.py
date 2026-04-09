import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# 1. LOAD DATASET
# header=None because your file starts immediately with numbers
# We assign names manually to match your screenshot
df = pd.read_csv("Gumel.csv", header=None)
df.columns = ["parameter", "year", "wind_speed", "temperature","pressure","precipitation","relative humidity", "cloud cover", "dewpoint","wind direction","Allsky-kt"]

# 2. SELECT FEATURES
# We use wind_speed and temperature since those are columns C and D
FEATURES = ["wind_speed", "temperature"]
data = df[FEATURES].values

# 3. SCALE DATA
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, time_steps=24, forecast_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_steps):
        X.append(data[i:i + time_steps])
        # Predict only wind speed (column index 0)
        y.append(data[i + time_steps:i + time_steps + forecast_steps, 0])
    return np.array(X), np.array(y)

TIME_STEPS = 24       # look back 24 rows
FORECAST_STEPS = 6    # predict next 6 rows

X, y = create_sequences(data_scaled, TIME_STEPS, FORECAST_STEPS)

# Split into Train and Test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. BUILD THE MODEL
model = Sequential([
    Bidirectional(
        LSTM(64, return_sequences=True), 
        input_shape=(TIME_STEPS, X.shape[2])
    ),
    LSTM(32),
    Dense(FORECAST_STEPS)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# 5. TRAIN
early_stop = EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 6. PREDICTIONS & INVERSE SCALING
predictions = model.predict(X_test)

def inverse_wind_speed(scaled_wind):
    # This must match the number of FEATURES (2)
    dummy = np.zeros((scaled_wind.shape[0], len(FEATURES)))
    dummy[:, 0] = scaled_wind
    return scaler.inverse_transform(dummy)[:, 0]

# Inverse transform for the first hour of the forecast
y_test_inv = inverse_wind_speed(y_test[:, 0])
pred_inv = inverse_wind_speed(predictions[:, 0])

# 7. EVALUATE
rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
mae = mean_absolute_error(y_test_inv, pred_inv)

print(f"\n--- Results ---")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")

# 8. PLOT
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="Actual Wind Speed", color='blue', alpha=0.7)
plt.plot(pred_inv, label="Predicted Wind Speed", color='red', linestyle='--')
plt.xlabel("Time Steps")
plt.ylabel("Wind Speed")
plt.title("Wind Speed Prediction (Test Set)")
plt.legend()
plt.show()

# 9. FORECAST FUTURE
last_sequence = data_scaled[-TIME_STEPS:]
last_sequence = last_sequence.reshape(1, TIME_STEPS, len(FEATURES))

future_pred = model.predict(last_sequence)
future_wind = inverse_wind_speed(future_pred[0])

print("\nPredicted wind speed for the next 6 steps:")
for i, val in enumerate(future_wind, 1):
    print(f"Step +{i}: {val:.2f}")                                                         