# ===============================
# MULTISTEP WIND SPEED FORECAST
# ===============================

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# ===============================
# PARAMETERS
# ===============================
look_back = 7        # past 7 days
horizon   = 3        # predict next 3 days
n_features = 2       # temperature + wind_speed


# ===============================
# LOAD SCALED DATA
# ===============================
X_train = np.load("X_train_multivariate.npy")
X_test  = np.load("X_test_multivariate.npy")
y_train = np.load("y_train_multivariate.npy")
y_test  = np.load("y_test_multivariate.npy")

scaler = joblib.load("multivariate_scaler.save")


# ===============================
# BUILD MULTI-STEP TARGET
# ===============================
def create_multistep(X, y_single, horizon):
    y_multi = []
    for i in range(len(y_single) - horizon + 1):
        y_multi.append(y_single[i:i+horizon])
    return np.array(y_multi)

# Rebuild multi-step y
y_train_multi = create_multistep(X_train, y_train, horizon)
y_test_multi  = create_multistep(X_test, y_test, horizon)

# Trim X to match new y size
X_train = X_train[:len(y_train_multi)]
X_test  = X_test[:len(y_test_multi)]


# ===============================
# MODEL ARCHITECTURE
# ===============================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(look_back, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(horizon)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

model.summary()


# ===============================
# CALLBACKS
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-5
)


# ===============================
# TRAINING
# ===============================
history = model.fit(
    X_train, y_train_multi,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# ===============================
# PREDICTION
# ===============================
y_pred = model.predict(X_test)


# ===============================
# INVERSE SCALING
# ===============================
def inverse_wind_only(scaled_values):
    dummy_temp = np.zeros_like(scaled_values)
    combined = np.stack([dummy_temp, scaled_values], axis=2)
    reshaped = combined.reshape(-1, 2)
    inv = scaler.inverse_transform(reshaped)
    return inv[:, 1].reshape(scaled_values.shape)

y_pred_inv = inverse_wind_only(y_pred)
y_test_inv = inverse_wind_only(y_test_multi)


# ===============================
# EVALUATION PER HORIZON
# ===============================
for h in range(horizon):
    rmse = np.sqrt(mean_squared_error(y_test_inv[:, h], y_pred_inv[:, h]))
    mae = mean_absolute_error(y_test_inv[:, h], y_pred_inv[:, h])
    print(f"Horizon t+{h+1} → RMSE: {rmse:.4f} | MAE: {mae:.4f}")


# ===============================
# PLOT FIRST HORIZON
# ===============================
plt.figure(figsize=(10,5))
plt.plot(y_test_inv[:,0], label="Actual t+1")
plt.plot(y_pred_inv[:,0], label="Predicted t+1")
plt.legend()
plt.title("Multi-step Wind Speed Forecast (t+1)")
plt.show()
