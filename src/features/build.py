from src.features.engineering import (
    create_lags, create_logreturns, create_temp_features,
    create_volume_features, create_dynamic_corr,
    create_vol_features, create_market_regimes,
    create_moving_averages, create_diffs
)
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

def build_all_features(
        df: pd.DataFrame,
        target_price_col: str,
        exog_price_cols: Optional[List[str]] = None,
        volume_col: Optional[str] = None,
        vix_col: str = '^VIX',
        econ_ind: Dict[str, int] = None,
        windows: List[int] = None,
        lags: List[int] = None
)-> pd.DataFrame:
    """
    Aplica TODAS as features ANTES do split.
    """
    df = df.copy()

    # Validações iniciais
    exog_price_cols = exog_price_cols or []
    windows = windows or [5, 22, 63]
    lags = lags or [1, 5, 22]

    # === VALIDAÇÃO: volume NUNCA é preço ===
    if volume_col and volume_col in exog_price_cols:
        raise ValueError(
            f"volume_col '{volume_col}' não pode estar em exog_price_cols. "
            "Volume não é preço → não deve ter log-retorno."
        )

    # 1. Log-retornos
    price_cols = [target_price_col] + (exog_price_cols or [])
    df = create_logreturns(df, price_cols)

    #renomeando coluna do ln ret do ativo alvo
    df = df.rename(columns={f"{target_price_col}_logreturns": "log_return"})

    # 2. Log-volume
    if volume_col is not None and volume_col in df.columns:
        df = df[df[volume_col] > 0]
        df['log_volume'] = np.log(df[volume_col])
        df = df.drop(columns=[volume_col])

    # 3. Lags (só em log-retornos e log-volume)
    lag_cols = [f'{col}_logreturns' for col in exog_price_cols]
    if 'log_volume' in df.columns:
        lag_cols.append('log_volume')
    lag_cols = [col for col in lag_cols if col in df.columns]
    df = create_lags(df, lag_cols, lags)
    
    # 4. Temporais
    df = create_temp_features(df)
    
    # 5. Volume features
    if 'log_volume' in df.columns:
        df = create_volume_features(df, 'log_volume')
    
    # 6. Vol features
    logreturn_cols = [f"{col}_logreturns" for col in exog_price_cols]
    logreturn_cols = [col for col in logreturn_cols if col in df.columns]
    df = create_vol_features(df, 'log_return', logreturn_cols, windows)
    
    # 7. Correlações dinâmicas
    df = create_dynamic_corr(df, 'log_return', logreturn_cols, windows)
    
    # 8. MAs
    ma_cols = ["log_return"] + logreturn_cols
    df = create_moving_averages(df, ma_cols, windows)
    
    # 9. Regimes
    if vix_col in df.columns:
        df = create_market_regimes(df, vix_col)
    
    # 10 Diffs
    if econ_ind:
        df = create_diffs(df, econ_ind, lags)
    
    # 11 Events
    if 'selic' in econ_ind:
        df['selic_event'] = (df['diff_1_selic'] != 0).astype(int)
        df.drop(columns = 'diff_1_selic',inplace=True)
        df = create_lags(df, 'selic_event', lags)

    df = df.dropna()
    df = df.reset_index()

    return df.reset_index(drop=True)