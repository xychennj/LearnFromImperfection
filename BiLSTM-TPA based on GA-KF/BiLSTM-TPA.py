import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import (
    Input, LSTM, Bidirectional, Dense,
    Conv1D, GlobalAveragePooling1D, Layer,
    concatenate, Activation, RepeatVector, Multiply, Lambda
)
from keras.models import Model
from keras.regularizers import l1_l2
from keras.callbacks import ReduceLROnPlateau
import os

INPUT_FILE_PATH = "*******"

OUTPUT_FILE_PATH = "*********"

SEQUENCE_LENGTH = 12
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.7

def temporal_pattern_attention(lstm_output, hidden_units, num_filters=32, filter_width=3):
    h_t = Lambda(lambda x: x[:, -1, :])(lstm_output)

    cnn_features = Conv1D(filters=num_filters,
                          kernel_size=filter_width,
                          padding='same',
                          activation='relu')(lstm_output)

    h_t_transformed = Dense(num_filters)(h_t)
    h_t_transformed = RepeatVector(cnn_features.shape[1])(h_t_transformed)

    attention_score = Multiply()([cnn_features, h_t_transformed])
    attention_score = Dense(1)(attention_score)

    attention_weights = Activation('sigmoid')(attention_score)

    context_vector = Multiply()([cnn_features, attention_weights])
    context_vector = GlobalAveragePooling1D()(context_vector)

    final_output = concatenate([h_t, context_vector])
    final_output = Dense(hidden_units, activation='relu')(final_output)

    return final_output

def calculate_safe_mape(y_true, y_pred, threshold=0.1):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = np.abs(y_true) > threshold

    if np.sum(mask) == 0:
        return 0.0

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return None, None

    print("Loading data...")
    df = pd.read_excel(filepath)

    required_columns = ['date', 'G4', 'water_level', 'rainfall']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file columns mismatch, required: {required_columns}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df[required_columns[1:]] = df[required_columns[1:]].interpolate(method='linear', limit_direction='both')

    dates = df['date'].values
    dataset = df[['G4', 'water_level', 'rainfall']].values

    return dates, dataset

def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def main():
    dates, raw_data = load_and_preprocess_data(INPUT_FILE_PATH)
    if dates is None: return

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler_X.fit_transform(raw_data)

    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    scaler_Y.fit(raw_data[:, 0].reshape(-1, 1))

    X, Y = create_dataset(data_normalized, SEQUENCE_LENGTH)
    prediction_dates = dates[SEQUENCE_LENGTH:]

    train_size = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)

    lstm_out = Bidirectional(LSTM(64, return_sequences=True,
                                  kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(inputs)

    tpa_out = temporal_pattern_attention(lstm_out, hidden_units=64)

    outputs = Dense(1)(tpa_out)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    print(f"Starting training (Total {EPOCHS} epochs)...")
    model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        verbose=1
    )

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict_orig = scaler_Y.inverse_transform(train_predict)
    test_predict_orig = scaler_Y.inverse_transform(test_predict)
    Y_train_orig = scaler_Y.inverse_transform(Y_train.reshape(-1, 1))
    Y_test_orig = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(Y_test_orig, test_predict_orig))
    r2 = r2_score(Y_test_orig, test_predict_orig)
    mape = calculate_safe_mape(Y_test_orig, test_predict_orig, threshold=0.1)

    print("\n" + "=" * 40)
    print(f"Evaluation Results (Test Set):")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.4f}% (Excluded small deformation data <0.1mm)")
    print("=" * 40 + "\n")

    full_predict = np.concatenate((train_predict_orig, test_predict_orig), axis=0)
    full_observed = np.concatenate((Y_train_orig, Y_test_orig), axis=0)

    result_df = pd.DataFrame({
        'Date': prediction_dates,
        'Observed': full_observed.flatten(),
        'Predicted': full_predict.flatten()
    })

    result_df['Metrics'] = np.nan
    result_df.loc[0, 'Metrics'] = f"R2: {r2:.4f}"
    result_df.loc[1, 'Metrics'] = f"RMSE: {rmse:.4f}"
    result_df.loc[2, 'Metrics'] = f"MAPE: {mape:.4f}"

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    result_df.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"Results saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()