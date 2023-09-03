"""
Created on Sat Sep  2 8:26:52 2023

@author: ryanm
"""

import numpy as np
import pandas as pd
import cvxpy as cvx
from datetime import date


from Project1.data import DataCollector

tickers = pd.read_csv('Project1/ticker_data.csv').Ticker
data = DataCollector(tickers, date(2020, 1, 1), date(2023, 1, 1))


class BmOptimizer:
    """Functional Class to estimate a benchmark portfolio given granular assets."""

    def __init__(self, benchmark, returns):
        self.benchmark = benchmark
        self.returns = returns

    def get_idx(self):
        """Return the index values of the benchmark and assets."""
        n = len(self.returns.columns)
        bm_idx = self.returns.columns.get_loc(self.benchmark)
        asset_idx = np.arange(n)[np.arange(n) != 1]
        return bm_idx, asset_idx, n

    def estimate_bm(self):
        """Returns estimate benchmark portfolio."""
        bm_idx, asset_idx, n = self.get_idx()

        portfolio = cvx.Variable(n, nonneg=True)
        benchmark = cvx.Variable(n, nonneg=True)

        constraints = [
            sum(portfolio) == 1,
            portfolio <= 1,
            portfolio[bm_idx] == 0,
            benchmark[asset_idx] == 0,
            benchmark[bm_idx] == 1
        ]

        objective = cvx.quad_form(portfolio - benchmark, self.returns.cov())

        cvx.Problem(cvx.Minimize(objective), constraints).solve()
        return pd.Series(portfolio.value, index=self.returns.columns)


if __name__ == '__main__':

    optimizer = BmOptimizer('IVV', data.returns)
    # print(optimizer.estimate_bm())
