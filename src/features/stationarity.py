from statsmodels.tsa.stattools import adfuller
import pandas as pd

def adf_series(df, names):
    '''
    df: DataFrame com as séries;

    names: lista com as séries a serem testadas

    output: DataFrame com resultados

    '''
    try:
        if isinstance(names,str):
            names = [names]
    except Exception as e:
        print(f'Erro com tipo de dados em {names}: {str(e)}.')

    results=[]
    for var in names:
        t,p = adfuller(df[var])[0:2]
        results.append([t,p])
    
    
    data={'info':['statistics','p-value','stationarity']}
    for j in range(len(names)):
        var = names[j]
        if results[j][1]<=0.05:
            status = 'stationary'
        else:
            status = 'non-stationary'
        results[j].append(status)
        data.update({var: results[j]})

    dt = pd.DataFrame(data)

    dt.set_index('info')

    return dt

    