import pandas as pd
import numpy as np
from itertools import combinations

def create_lags(df, tickers, lags):
    """
    Cria lags para as features selecionadas.

    Parâmetros:
    -----------
    df: DataFrame
        DataFrame com os dados que vamos operar
    tickers : str ou list
        Ticker ou lista de tickers para download (ex: 'PETR4.SA' ou ['PETR4.SA', 'VALE3.SA'])
    lags: list
        Lista com inteiros que serão os lags criados
    
    Retorna:
    --------
    df_copy: DataFrame com os lags incluídos.
    
    """
    df_copy = df.copy()

    try:
        if isinstance(tickers,str):
            ticker = [tickers]
        elif isinstance(tickers,list):
            ticker = tickers
    except Exception as e:
        print(f'Erro com tipo de dado em {tickers}: {str(e)}.')
    
    if isinstance(lags,list):
        if len(lags)>0:
            for col in ticker:
                for lag in lags:
                    df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        else:
            print('Insira uma lista não nula de lags (inteiros).')
    
    return df_copy



def create_logreturns(df,tickers):
    """
    Cria os log retornos das variáveis selecionadas em tickers
    
    Parâmetros:
    -----------
    df: DataFrame
        DataFrame com os dados que vamos operar
    tickers : str ou list
        Ticker ou lista de tickers (ex: 'PETR4.SA' ou ['PETR4.SA', 'VALE3.SA'])
    
    Retorna:
    --------
    
    df_copy: DataFrame com os log retornos incluídos.
    
    """
    
    df_copy = df.copy()

    try:
        if isinstance(tickers,str):
            ticker = [tickers]
        elif isinstance(tickers,list):
            ticker = tickers
    except Exception as e:
        print(f'Erro com tipo de dado em {tickers}: {str(e)}.')
        
    for col in ticker:
        if (np.any(df_copy[col].isnull()) or np.any(df_copy[col]==0)):
            raise ValueError(f'A coluna {col} apresenta zeros ou nulos.')
        else:
            df_copy[f'{col}_logreturns'] = np.log(df_copy[col] / df_copy[col].shift(1))
            
    return df_copy

def create_temp_features(df):
    """
    Parâmetros:
    ----------
    df: DataFrame
        DataFrame com os dados que vamos operar
    
    Retorna:
    --------
    
    df_copy: DataFrame com as features.
    """
    df_copy = df.copy()
    
    df_copy['month'] = df.index.month
    df_copy['weekday'] = df.index.weekday
    df_copy['quarter'] = df.index.quarter
    df_copy['is_month_end'] = df_copy.index.is_month_end.astype(int)
    df_copy['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df_copy



def create_volume_features(df,log_volume_col):
    
    """
    Parâmetros:
    ----------
    df: DataFrame
        DataFrame com os dados que vamos operar
    log_volume_col: str 

    As médias trabalhadas aqui são EWMA
    
    Retorna:
    --------
    
    df_copy: DataFrame com as features.
    """
    df_copy = df.copy()
    windows=[5,21]
    momentum_windows = [3,5]

    # Validando o input
    if log_volume_col not in df_copy.columns:
        raise ValueError(f"A coluna de log-volume '{log_volume_col}' não foi encontrada no DataFrame.")

    # Variação do log volume
    log_volume_change_col = f'{log_volume_col}_diff_1'
    df_copy[log_volume_change_col] = df_copy[log_volume_col].diff()

    # Médias Móveis (EWMA)
    for w in windows:
        df_copy[f'{log_volume_col}_ewm_{w}'] = df_copy[log_volume_col].ewm(span=w).mean()

    # Volume 'Buzz'
    # esta feature mede o quanto o volume atual muda de acordo com a média EWMA de período w
    for w in windows:
        df_copy[f'{log_volume_col}_buzz_{w}'] = (df_copy[log_volume_col] - df_copy[f'{log_volume_col}_ewm_{w}']) # log(a) - log(b) = log(a/b)

    # Momentum do Volume
    # esta feature ajuda a verificar se o volume está aumentando ou reduzindo consistentemente
    for w in momentum_windows:
        df_copy[f'{log_volume_col}_momentum_{w}'] = df_copy[log_volume_change_col].rolling(w).sum()

    # Volatilidade do Volume
    for w in windows:
        df_copy[f'{log_volume_col}_volatility_{w}'] = df_copy[log_volume_change_col].rolling(w).std()

    # Pico de Volume
    df_copy['volume_spike'] = (df_copy[f'{log_volume_col}_buzz_{max(windows)}'] > np.log(2.0)).astype(int) # o volume de hoje é mais que o dobro da média móvel recente?? Isso que esta feature mostra
    
    return df_copy



def create_dynamic_corr(df,feature_principal,features,windows):
    
    df_copy=df.copy()
    
    if isinstance(feature_principal, str):
        if len(feature_principal)==0:
            raise ValueError('A variável principal não deve ser uma string vazia')
        ticker = [feature_principal]
    elif isinstance(feature_principal, list):
        if len(feature_principal)==0:
            raise ValueError('A variável principal não deve ser uma lista vazia')
        ticker = feature_principal
    
    if isinstance(features, str):
        if len(features)==0:
            raise ValueError('As features não devem ser uma string vazia')
        feat = [features]
    elif isinstance(features, list):
        if len(features)==0:
            raise ValueError('As features não devem ser uma lista vazia')
        feat = features
    
    # Correlações dinâmicas entre feature_principal e outras features
    for f in feat:
        for w in windows:
            df_copy[f'corr_{ticker[0]}_{f}_{w}'] = df_copy[ticker[0]].rolling(w).corr(df_copy[f])
    
    # Correlações entre os ativos em features:
    combinacoes = [list(x) for x in list(combinations(feat,2))]
    
    for comb in combinacoes:
        for w in windows:
            df_copy[f'corr_{comb[0]}_{comb[1]}_{w}'] = df_copy[comb[0]].rolling(w).corr(df_copy[comb[1]])
    
    return df_copy



def create_vol_features(df, feature_principal, features, windows):
    df_copy = df.copy()
    
    # Normalizar feature_principal para lista
    if isinstance(feature_principal, str):
        if len(feature_principal) == 0:
            raise ValueError('A variável principal não deve ser uma string vazia')
        ticker = [feature_principal]
    elif isinstance(feature_principal, list):
        if len(feature_principal) == 0:
            raise ValueError('A variável principal não deve ser uma lista vazia')
        ticker = feature_principal
    
    # Normalizar features para lista
    if isinstance(features, str):
        if len(features) == 0:
            raise ValueError('As features não devem ser uma string vazia')
        feat = [features]
    elif isinstance(features, list):
        if len(features) == 0:
            raise ValueError('As features não devem ser uma lista vazia')
        feat = features
        
    # Validar se colunas existem
    all_columns = [ticker[0]] + feat
    missing_cols = [col for col in all_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Colunas não encontradas no DataFrame: {missing_cols}')
        
    
    ## 1. Vol do ativo principal
    #for w in windows:
     #   df_copy[f'{ticker[0]}_vol_{w}'] = df[ticker[0]].rolling(w).std()
        
    ## 2. Vol das exógenas
    #for f in feat:
     #   for w in windows:
      #    df_copy[f'{f}_vol_{w}'] = df[f].rolling(w).std()

    # 1 Vol do ativo principal e exógenas
    for asset in all_columns:
        for w in windows:
            df_copy[f'{asset}_vol_{w}'] = df[asset].ewm(span=w, min_periods=w).std() * np.sqrt(252)
    
    # 2. Razões de vol (Regime Detection)
    if len(windows) >= 2:
        # Ratio do ativo principal (curto/longo prazo)
        short_window = min(windows)
        long_window = max(windows)
        df_copy[f'{ticker[0]}_vol_ratio_{short_window}_{long_window}'] = (
            df_copy[f'{ticker[0]}_vol_{short_window}'] / df_copy[f'{ticker[0]}_vol_{long_window}'])
        
        # Ratios das exógenas
        for f in feat:
            df_copy[f'{f}_vol_ratio_{short_window}_{long_window}'] = (
                df_copy[f'{f}_vol_{short_window}'] / df_copy[f'{f}_vol_{long_window}'])
            
    # 3. Vol relativa (spreads)
    for f in feat:
        for w in windows:
            df_copy[f'vol_spread_{ticker[0]}_{f}_{w}'] = (
                df_copy[f'{ticker[0]}_vol_{w}'] - df_copy[f'{f}_vol_{w}'])
            
    
    # 4. Vol correlations
    if len(windows) >= 1:
        vol_window = max(windows)
        for f in feat:
            df_copy[f'vol_corr_{ticker[0]}_{f}_{vol_window}'] = (
                df_copy[f'{ticker[0]}_vol_{vol_window}'].rolling(vol_window).corr(df_copy[f'{f}_vol_{vol_window}']))
    
    # 5. Regimes de volatilidade (percentis fixos )
    for asset in all_columns:
        if len(windows) >= 1:
            long_window = max(windows)
            vol_column = f'{asset}_vol_{long_window}'
            rolling_quantile_window = 252 # Usa 1 ano de dados para definir o que é "alto/baixo"
            
            # Percentis dinâmicos
            vol_75th_rolling = df_copy[vol_column].rolling(window=rolling_quantile_window, min_periods=int(rolling_quantile_window*0.8)).quantile(0.75) # 0.8 indica que podemos executar o cálculo com 80% dos dados necessários
            vol_25th_rolling = df_copy[vol_column].rolling(window=rolling_quantile_window, min_periods=int(rolling_quantile_window*0.8)).quantile(0.25)
            
            df_copy[f'{asset}_high_vol_regime'] = (df_copy[vol_column] > vol_75th_rolling).astype(int)
            df_copy[f'{asset}_low_vol_regime'] = (df_copy[vol_column] < vol_25th_rolling).astype(int)
            
            
    return df_copy       

def create_market_regimes(df, vix_col='^VIX',vix_logret_col='VIX_logreturns'):
    
    df_copy = df.copy()
    
    # Regimes de stress
    df_copy['vix_regime_low'] = (df[vix_col] < 15).astype(int)
    df_copy['vix_regime_high'] = (df[vix_col] > 25).astype(int)
    
    # Mudanças bruscas
    df_copy['vix_spike'] = (df[vix_col].pct_change() > 0.2).astype(int)    
    df_copy['vix_calm_down'] = (df[vix_logret_col] < -0.15).astype(int)
    
    return df_copy



def create_moving_averages(df,tickers,windows):
    """
    Cria médias móveis básicas
    
    Parâmetros:
    -----------
    df: DataFrame
        DataFrame com os dados
    tickers: str ou list
        Ticker(s) para calcular médias móveis
    windows: list
        Lista de janelas para as médias (ex: [5, 22, 63])
    
    Retorna:
    --------
    df_copy: DataFrame com médias móveis
    """
    df_copy = df.copy()
    
    # Normalizar tickers para lista
    if isinstance(tickers, str):
        if len(tickers) == 0:
            raise ValueError('Ticker não deve ser string vazia')
        ticker = [tickers]
    elif isinstance(tickers, list):
        if len(tickers) == 0:
            raise ValueError('Lista de tickers não deve ser vazia')
        ticker = tickers
    
    # Validar se colunas existem
    missing_cols = [col for col in ticker if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Colunas não encontradas: {missing_cols}')
        
    # Médias Móveis básicas
    for col in ticker:
        for w in windows:
            df_copy[f'{col}_ma_{w}'] = df[col].rolling(w).mean()
    
    # Posição relativa
    for col in ticker:
        for w in windows:
            df_copy[f'{col}_above_ma_{w}'] = (df[col] > df_copy[f'{col}_ma_{w}']).astype(int) 
    
    # Spread entre média curta e longa
    if len(windows)>=2:
        
        curta = min(windows)
        longa = max(windows)
        
        for col in ticker:
            df_copy[f'{col}_spread_ma_{curta}_{longa}'] = df_copy[f'{col}_ma_{curta}'] - df_copy[f'{col}_ma_{longa}']
        
    
    return df_copy

def create_diffs(df, variables, lags):
    df_copy = df.copy()

    if isinstance(variables, list):
        if len(variables)==0:
            raise ValueError('Lista de variáveis vazia.')
    elif isinstance(variables, str):
        if len(variables)==0:
            raise ValueError('Variável não deve ser string vazia')
        variables = [variables]

    for var in variables:
        df_copy[f'diff_{var}'] = df_copy[var].diff(1)
    
    return df_copy