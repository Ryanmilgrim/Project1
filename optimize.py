"""
Created on Sun Sep  3 14:42:37 2023

@author: ryanm
"""

import numpy as np
import pandas as pd
import cvxpy as cvx
from datetime import date

from Project1.util.data import DataCollector


class BmOptimizer:
    """Functional Class to estimate a benchmark portfolio given granular assets."""

    def __init__(self, benchmark, returns):
        # Parsing useful data from input variables. Also creating a benchmark vector.
        n_periods, n_assets = returns.shape
        bm_asset = returns.columns.isin(benchmark.index)

        # Setting up the portfolio Variable, returns Parameter, and Constant BM Vector.
        self.portfolio = cvx.Variable(n_assets, nonneg=True)
        self.returns = cvx.Parameter(returns.shape)
        self.benchmark = pd.Series(benchmark, index=returns.columns).replace(np.nan, 0)

        # Defining optimization objectives and constraints.
        self.objective = cvx.sum_squares(
            (self.portfolio - self.benchmark) @ self.returns.T)
        self.constraints = [
            self.portfolio <= 1,  # No Weights greater than 100%.
            self.portfolio[bm_asset] == 0,  # Benchmark assets are not investable.
            sum(self.portfolio) == 1  # Portfolio must be fully allocated.
        ]

        # Assigning the parameterized optimization.
        self.problem = cvx.Problem(cvx.Minimize(self.objective), self.constraints)

    def estimate_bm(self, returns):
        """Returns estimate benchmark portfolio."""
        self.returns.value = returns.to_numpy()
        self.problem.solve()
        return pd.Series(self.portfolio.value, index=returns.columns)


if __name__ == '__main__':
    tickers = pd.read_csv('Project1/util/ticker_data.csv').Ticker
    data = DataCollector(tickers, date(2020, 1, 1), date(2023, 1, 1))
    benchmark = pd.Series({'IVV': 0.9, 'IYK': 0.1})
    estimate = BmOptimizer(benchmark, data.returns).estimate_bm(data.returns)
    print(estimate.round(6))
