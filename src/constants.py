# src/constants.py

# ==== DADOS ====
TICKERS = ["PETR4.SA", "^BVSP", "CL=F", "BRL=X"]
START_DATE = "2015-01-01"
END_DATE = "2025-11-13"

# ==== FEATURES ====
USE_LOG_RETURNS = True
LAGS = [1, 5, 22]  # lags para variáveis exógenas

# ==== MODELO ARIMAX ====
PMDARIMA_AUTO = True  # <--- AQUI É O QUE VOCÊ QUER!
# Se False, usa: ARIMAX_ORDER = (2, 0, 1)

# ==== MODELO XGBoost ====
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42
}

# ==== VALIDAÇÃO ====
TRAIN_SIZE = 0.8