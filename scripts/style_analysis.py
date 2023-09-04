"""
Created on Sun Sep  3 20:09:02 2023

@author: ryanm
"""

import pandas as pd
from datetime import date

from Project1.optimize import BmOptimizer
from Project1.util.data import DataCollector


def StyleAnalysis(benchmark, trailing_period=60, years=10):

    # Ensure benchmark is formatted into a pd.series
    if isinstance(benchmark, str):
        benchmark = pd.Series({benchmark: 1.0})

    # Date and Ticker setup.
    end = date.today()
    start = end.replace(year=end.year - years)
    tickers = pd.read_csv('Project1/util/ticker_data.csv').Ticker

    # Performing API Call.
    data = DataCollector(tickers, start, end)

    # Looping through historical returns and recording allocations.
    allocations = dict()
    for i, period in enumerate(data.returns.index):

        # Only optimize on periods which satisfy the minimum window size.
        if i < trailing_period:
            continue

        # Creating a slice of history for optimization.
        returns = data.returns[i - trailing_period: i]

        # Once min window is satisfied, the optimizer is assigned and used.
        if i == trailing_period:
            optimizer = BmOptimizer(benchmark, returns)
        allocations[period] = optimizer.estimate_bm(returns)

    allocations = pd.DataFrame().from_dict(allocations)
    allocations = allocations.T.sort_index()
    return allocations


if __name__ == '__main__':
    benchmark = pd.Series({'IVV': 1.0})
    results = main(benchmark, 90)
    results.plot(kind='area')
