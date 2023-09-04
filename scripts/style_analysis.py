"""
Created on Sun Sep  3 20:09:02 2023

@author: ryanm
"""

import pandas as pd
from Project1.optimize import BmOptimizer


def StyleAnalysis(benchmark, returns, trailing_period=60):

    # Ensure benchmark is formatted into a pd.series
    if isinstance(benchmark, str):
        benchmark = pd.Series({benchmark: 1.0})

    # Looping through historical returns and record allocations.
    allocations = dict()
    for i, period in enumerate(returns.index):

        # Only optimize on periods which satisfy the minimum window size.
        if i < trailing_period:
            continue

        # Creating a slice of history for optimization.
        return_subset = returns[i - trailing_period: i]

        # Once min window is satisfied, the optimizer is assigned and used.
        if i == trailing_period:
            optimizer = BmOptimizer(benchmark, return_subset)
        allocations[period] = optimizer.estimate_bm(return_subset)

    # Wrangling outputs neatly.
    allocations = pd.DataFrame().from_dict(allocations)
    allocations = allocations.T.sort_index()
    return allocations
