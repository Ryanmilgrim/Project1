"""
Created on Sat Sep  2 8:26:52 2023

@author: ryanm
"""

import numpy as np
import pandas as pd
import cvxpy as cvx
from datetime import date


from Project1.data import DataCollector


df = pd.read_csv('Project1/ticker_data.csv', index_col=0)
data = DataCollector(df.index, date(2020, 1, 1), date(2023, 1, 1))

# %%


class BmOptimizer:
    """Functional Class to estimate a benchmark portfolio given granular assets."""

    def __init__(self, benchmark, returns):
        # Parsing useful data from input variables.
        n_periods, n_assets = returns.shape
        self.bm_asset = returns.columns.isin(benchmark.index)

        # Setting up the portfolio Variable, Cov Parameter, and Constant BM Vector.
        self.portfolio = cvx.Variable(n_assets)
        self.returns = cvx.Parameter(returns.shape)

        # Defining optimization objectives and constraints.
        self.objective = cvx.sum_squares(self.portfolio @ self.returns.T)
        self.constraints = [
            self.portfolio <= 1,  # No Weights greater than 100%.
            self.portfolio[self.bm_asset] == -1,  # Benchmark assets are not investable.
            self.portfolio[~self.bm_asset] >= 0,  # Benchmark assets are not investable.
            sum(self.portfolio) == 0  # Portfolio must be fully allocated.
        ]

        # Assigning the parameterized optimization.
        self.problem = cvx.Problem(cvx.Minimize(self.objective), self.constraints)

    def estimate_bm(self, returns):
        """Returns estimate benchmark portfolio."""
        self.returns.value = returns.to_numpy()
        self.problem.solve()
        result = pd.Series(self.portfolio.value, index=returns.columns).clip(0, 1)
        result[self.bm_asset] = 0
        return result


if __name__ == '__main__':

    benchmark = pd.Series({'IVV': 1.0})
    optimizer = BmOptimizer(benchmark, data.returns)
    print(optimizer.estimate_bm(data.returns))
