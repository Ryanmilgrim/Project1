"""
Created on Sat Sep  2 8:26:52 2023

@author: ryanm
"""

import pandas as pd
from datetime import date

from Project1.data import DataCollector
from Project1.optimize import BmOptimizer


def main():
    tickers = pd.read_csv('Project1/ticker_data.csv').Ticker
    data = DataCollector(tickers, date(2020, 1, 1), date(2023, 1, 1))
    benchmark = pd.Series({'IVV': 1.0})
    optimizer = BmOptimizer(benchmark, data.returns)
    print(optimizer.estimate_bm())


if __name__ == '__main__':
    main()
