import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
except ImportError:
    EarlyStopping = None
    Dense = None
    Dropout = None
    LSTM = None
    Sequential = None


ALL_FEATURE_COLUMNS = [
    "brent_price",
    "policy_rate",
    "general_cpi",
    "food_cpi",
    "bdi_mean",
    "bdi_std",
    "bdi_range",
    "bdi_cv",
    "bdi_return",
    "bdi_month_direction",
    "bdi_extreme_days",
    "bdi_rolling_mean_3",
    "bdi_rolling_std_3",
    "brent_price_lag_1",
    "brent_price_lag_3",
    "brent_price_lag_6",
    "brent_price_lag_12",
    "policy_rate_lag_1",
    "policy_rate_lag_3",
    "policy_rate_lag_6",
    "policy_rate_lag_12",
    "bdi_mean_lag_1",
    "bdi_mean_lag_3",
    "bdi_mean_lag_6",
    "bdi_mean_lag_12",
    "food_price_inflation_lag_1",
    "food_price_inflation_lag_3",
    "food_price_inflation_lag_6",
    "food_price_inflation_lag_12",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the food inflation forecasting baselines")
    parser.add_argument(
        "--driver",
        choices=["all", "bdi", "brent"],
        default="all",
        help="Feature focus for the training run: all, bdi, or brent.",
    )
    return parser.parse_args()


def get_feature_columns(driver_name: str):
    if driver_name == "bdi":
        return [
            column
            for column in ALL_FEATURE_COLUMNS
            if column.startswith("bdi_") or column in {"policy_rate", "general_cpi", "food_cpi"}
        ]
    if driver_name == "brent":
        return [
            column
            for column in ALL_FEATURE_COLUMNS
            if column.startswith("brent_price") or column in {"policy_rate", "general_cpi", "food_cpi"}
        ]
    return ALL_FEATURE_COLUMNS


def train_sarimax(train_data, test_data, driver_name="all"):
    feature_columns = get_feature_columns(driver_name)

    adf_statistic, adf_pvalue, *_ = adfuller(train_data["food_price_inflation"], autolag="AIC")

    model = SARIMAX(
        train_data["food_price_inflation"],
        exog=train_data[feature_columns],
        order=(1, 0, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=len(test_data), exog=test_data[feature_columns])
    predictions = forecast.predicted_mean
    actual = test_data["food_price_inflation"]
    residuals = actual - predictions

    return {
        "predictions": predictions,
        "actual": actual,
        "diagnostics": {
            "adf_statistic": adf_statistic,
            "adf_pvalue": adf_pvalue,
            "residual_mean": residuals.mean(),
            "residual_autocorr_1": residuals.autocorr(lag=1),
        },
        "metrics": {
            "mae": mean_absolute_error(actual, predictions),
            "rmse": mean_squared_error(actual, predictions) ** 0.5,
            "mape": mean_absolute_percentage_error(actual, predictions) * 100,
        },
    }


def train_lstm(train_data, test_data, lookback=12, driver_name="all"):
    feature_columns = get_feature_columns(driver_name)

    train_features = train_data[feature_columns].to_numpy()
    train_target = train_data["food_price_inflation"].to_numpy()
    test_features = test_data[feature_columns].to_numpy()
    test_target = test_data["food_price_inflation"].to_numpy()

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)

    # Helper: turn sliding windows into (X, y) pairs
    def make_sequences(values, target_values, lookback_value):
        X, y = [], []
        for i in range(len(values) - lookback_value + 1):
            X.append(values[i : i + lookback_value])
            y.append(target_values[i + lookback_value - 1])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(scaled_train, train_target, lookback)

    # Build test sequences by prepending the last `lookback` training rows
    # so we can produce one prediction for every test month.
    combined_values = np.concatenate([scaled_train[-lookback:], scaled_test], axis=0)
    combined_targets = np.concatenate([train_target[-lookback:], test_target], axis=0)
    X_test_full, y_test_full = make_sequences(combined_values, combined_targets, lookback)
    # Keep only the sequences that align with the test period
    X_test = X_test_full[-len(test_data):]
    y_test = y_test_full[-len(test_data):]

    if Sequential is None:
        predictions = np.zeros(len(X_test))
        return {
            "predictions": predictions,
            "actual": y_test,
            "metrics": {
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": mean_squared_error(y_test, predictions) ** 0.5,
                "mape": mean_absolute_percentage_error(y_test, predictions) * 100,
            },
        }

    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = model.predict(X_test, verbose=0).ravel()

    return {
        "predictions": predictions,
        "actual": y_test,
        "metrics": {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": mean_squared_error(y_test, predictions) ** 0.5,
            "mape": mean_absolute_percentage_error(y_test, predictions) * 100,
        },
    }


if __name__ == "__main__":
    args = parse_args()
    from preprocessing import create_lag_features, load_data, split_train_test

    data = load_data()
    features = create_lag_features(data)
    train, test = split_train_test(features)
    print("driver_focus", args.driver)
    sarimax_result = train_sarimax(train, test, driver_name=args.driver)
    lstm_result = train_lstm(train, test, driver_name=args.driver)
    print(sarimax_result["metrics"])
    print(lstm_result["metrics"])
