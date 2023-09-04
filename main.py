"""
Created on Sat Sep  2 8:26:52 2023

@author: ryanm
"""

import pandas as pd
from datetime import date

from Project1.util.data import DataCollector
from Project1.scripts.style_analysis import StyleAnalysis

# %% 

end = date.today()
start = end.replace(year=end.year - 10)
tickers = pd.read_csv('Project1/util/ticker_data.csv').Ticker
data = DataCollector(tickers, start, end)

# %%


results = StyleAnalysis('IVV', data.returns, 90)
results.plot(kind='area')

