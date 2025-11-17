import numpy as np
import matplotlib.dates as mdates

def create_smart_date_formatter(dmin, dmax):
    """
    Fábrica que cria uma função de formatação de data inteligente.
    - Mostra a data completa no início e no fim.
    - Mostra apenas o ano para os ticks intermediários.
    - Suprime o tick de ano se ele for do mesmo ano que o início ou o fim.
    """
    dmin_num = mdates.date2num(dmin)
    dmax_num = mdates.date2num(dmax)

    def formatter(x, pos):
        """A função formatadora que será usada pelo Matplotlib."""
        dt = mdates.num2date(x)
        
        # 1. Se for (muito perto de) a data de início ou fim, formate por completo.
        if np.isclose(x, dmin_num) or np.isclose(x, dmax_num):
            return dt.strftime('%d/%m/%Y')

        # 2. (NOVA LÓGICA) Se o ano do tick for o mesmo do início ou do fim,
        #    mas não for o tick exato, retorne uma string vazia para escondê-lo.
        if dt.year == dmin.year or dt.year == dmax.year:
            return ''

        # 3. Para todos os outros casos (anos intermediários), mostre apenas o ano.
        else:
            return dt.strftime('%Y')

    return formatter