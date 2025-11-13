import pandas as pd
import os
from typing import Dict, Optional
from src.data.download import download_data
from src.constants import TICKERS


def build_main_dataset(
    target_ticker: str,
    target_name: str,
    aux_tickers: Dict[str, str],  # ex: {'^BVSP': 'ibov', 'CL=F': 'petroleo'}
    start: Optional[str] = None,
    end: Optional[str] = None,
    dir: str = 'data/raw',
    indicadores_bcb: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Junta ativo principal + exógenas + BCB.
    Salva em data/processed/
    """
    all_tickers = [target_ticker] + list(aux_tickers.keys())
    data_dict = download_data(all_tickers, start, end, dir=dir)

    # Ativo principal
    df = data_dict[target_ticker][['Adj Close', 'Volume']].copy()
    df = df[df['Volume'] > 0]
    df = df.rename(columns={'Adj Close': target_name})

    # Exógenas
    for ticker, name in aux_tickers.items():
        try:
            series = data_dict[ticker]['Adj Close'].rename(name)
            df = df.join(series, how='inner')
        except Exception as e:
            print(f"Erro com {ticker}: {e}")

    # BCB (se tiver)
    if indicadores_bcb:
        from bcb import sgs
        for nome, codigo in indicadores_bcb.items():
            try:
                series = sgs.get({nome: codigo}, start=start, end=end)
                if nome.lower() == 'selic':
                    series_aligned = series.reindex(df.index, method='ffill')
                else:
                    series_monthly = series.resample('M').last()
                    series_aligned = series_monthly.reindex(df.index, method='ffill')
                df = df.join(series_aligned, how='left')
            except Exception as e:
                print(f"Erro BCB {nome}: {e}")

    df = df.dropna()
    df.index.name = 'Date'

    # Salvar
    os.makedirs('data/processed', exist_ok=True)
    file_path = f"data/processed/{target_name}_completo.csv"
    df.to_csv(file_path)
    print(f"Dataset final salvo em {file_path}")

    return df