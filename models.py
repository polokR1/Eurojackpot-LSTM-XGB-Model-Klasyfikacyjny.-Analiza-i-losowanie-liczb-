# models.py
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, hamming_loss, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
import joblib
import os

def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def normalize_data(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_shape = X.shape
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)
    y_scaled = scaler_y.fit_transform(y)
    return X_scaled, y_scaled, scaler_X, scaler_y

def prepare_data_regression(data, lstm=False):
    main_numbers = data[['L1', 'L2', 'L3', 'L4', 'L5']].values
    euro_numbers = data[['L6', 'L7']].values
    additional_features = data[['Sum_Main', 'Mean_Main', 'Std_Main']].values
    X, y = [], []

    if lstm:
        window_size = 5
        num_features = 5 + 2 + 3  # Główne liczby, Euronumery, dodatkowe cechy
        for i in range(len(main_numbers) - window_size):
            X_sample_main = main_numbers[i:i + window_size]
            X_sample_euro = euro_numbers[i:i + window_size]
            X_sample_features = additional_features[i:i + window_size]
            X_sample = np.hstack((X_sample_main, X_sample_euro, X_sample_features))
            X.append(X_sample)
            y_sample = np.hstack((main_numbers[i + window_size], euro_numbers[i + window_size]))
            y.append(y_sample)
        X = np.array(X)  # Kształt: (liczba_próbek, window_size, num_features)
        y = np.array(y)  # Kształt: (liczba_próbek, 7)
    else:
        for i in range(len(main_numbers) - 1):
            X_sample = np.hstack((
                main_numbers[i],
                euro_numbers[i],
                additional_features[i]
            ))
            X.append(X_sample)
            y_sample = np.hstack((main_numbers[i + 1], euro_numbers[i + 1]))
            y.append(y_sample)
        X = np.array(X)
        y = np.array(y)

    return X, y

def prepare_data_classification(data):
    main_numbers = data[['L1', 'L2', 'L3', 'L4', 'L5']].values
    euro_numbers = data[['L6', 'L7']].values
    X, y = [], []
    for i in range(len(main_numbers) - 5):
        X_sample = np.hstack((
            main_numbers[i:i+5].flatten(),
            euro_numbers[i:i+5].flatten()
        ))
        y_sample_main = np.zeros(50)
        y_sample_euro = np.zeros(12)
        for num in main_numbers[i+5]:
            y_sample_main[int(num)-1] = 1
        for num in euro_numbers[i+5]:
            y_sample_euro[int(num)-1] = 1
        y_sample = np.hstack((y_sample_main, y_sample_euro))
        X.append(X_sample)
        y.append(y_sample)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_xgb_model(X, y, seed, params):
    set_random_seed(seed)
    X_scaled, y_scaled, scaler_X, scaler_y = normalize_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

    n_samples = X_train.shape[0]
    if n_samples < 5:
        raise ValueError(f"Zbyt mało danych w zbiorze treningowym ({n_samples} próbek). Wymagane co najmniej 5 próbek.")

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=seed,
        **params
    )
    xgb_reg.fit(X_train, y_train)

    # Kroswalidacja
    kf = KFold(n_splits=min(5, n_samples))
    scores = cross_val_score(xgb_reg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -scores
    average_mse = mse_scores.mean()

    # Evaluate the model
    predictions = xgb_reg.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return xgb_reg, scaler_X, scaler_y, mse, average_mse

def save_xgb_model(model, scaler_X, scaler_y, filepath):
    joblib.dump({'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, filepath)

def load_xgb_model(filepath):
    if os.path.exists(filepath):
        data = joblib.load(filepath)
        return data['model'], data['scaler_X'], data['scaler_y']
    else:
        return None, None, None

def train_lstm_model(X, y, seed, params):
    set_random_seed(seed)
    X_scaled, y_scaled, scaler_X, scaler_y = normalize_data(X, y)

    n_samples = X_scaled.shape[0]
    if n_samples < 5:
        raise ValueError(f"Zbyt mało danych do trenowania modelu LSTM ({n_samples} próbek). Wymagane co najmniej 5 próbek.")

    # Definicja modelu LSTM
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')

    # Trening modelu LSTM
    model.fit(X_scaled, y_scaled, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

    return model, scaler_X, scaler_y

def save_lstm_model(model, scaler_X, scaler_y, filepath):
    model.save(filepath)
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, filepath + '_scalers.pkl')

def load_lstm_model(filepath):
    if os.path.exists(filepath):
        model = load_model(filepath)
        scalers = joblib.load(filepath + '_scalers.pkl')
        return model, scalers['scaler_X'], scalers['scaler_y']
    else:
        return None, None, None

def train_classification_model(X, y, seed):
    set_random_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    n_samples = X_train.shape[0]
    if n_samples < 5:
        raise ValueError(f"Zbyt mało danych w zbiorze treningowym ({n_samples} próbek). Wymagane co najmniej 5 próbek.")

    base_clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
    clf = MultiOutputClassifier(base_clf)
    clf.fit(X_train, y_train)

    # Ocena modelu
    y_pred_test = clf.predict(X_test)
    test_hamming = hamming_loss(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='micro')

    return clf, test_hamming, f1

def save_classification_model(model, filepath):
    joblib.dump(model, filepath)

def load_classification_model(filepath):
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        return model
    else:
        return None
