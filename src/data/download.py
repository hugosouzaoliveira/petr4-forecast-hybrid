import yfinance as yf
import pandas as pd
import os
from typing import List, Dict, Optional
from src.constants import TICKERS, START_DATE, END_DATE

def download_data(tickers: List[str],
                  start: Optional[str] = None,
                  end: Optional[str] = None,
                  interval: str = '1d',
                  dir: str = 'data/raw',
                  auto_adjust: bool = False,
                  progress: bool = False) -> Dict[str, pd.DataFrame]:
    '''
    Baixa os dados do yfinance para vários tickers.
    Salva em data/raw/ se dir for informado.
    '''

    start = start or START_DATE
    end = end or END_DATE

    if isinstance(tickers, str):
        tickers = [tickers]

    os.makedirs(dir, exist_ok=True)
    result_dfs = {}

    for ticker in tickers:
        file_name = f"{ticker}_{start}_{end}.csv".replace('^', '').replace('=', '_')
        file_path = os.path.join(dir, file_name)

        try:
            if os.path.exists(file_path):
                print(f"Carregando {ticker} de {file_path}")
                data = pd.read_csv(file_path, parse_dates=['Date'])
                data.set_index('Date', inplace=True)
            else:
                print(f"Baixando {ticker}...")
                data = yf.download(
                    ticker, start=start, end=end, interval=interval,
                    auto_adjust=auto_adjust, progress=progress
                )
                if data.empty:
                    raise ValueError("Nenhum dado retornado")
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.to_csv(file_path)
                print(f"Salvou em {file_path}")

            result_dfs[ticker] = data

        except Exception as e:
            print(f"Erro em {ticker}: {e}")

    return result_dfs