"""
Created on Sat Sep  2 18:26:11 2023

@author: ryanm
"""

import pandas as pd
import yfinance as yf


class DataCollector:
    """Base Class for holding and collecting historical data."""

    def __init__(self, tickers, start, end, freq='1d'):
        self.tickers = tickers
        self.start, self.end, self.freq = start, end, freq
        self.init_data()

    def init_data(self):
        """Init main sources of data and stores extra data fields."""
        if isinstance(self.tickers, (pd.Index, pd.Series)):
            self.tickers = self.tickers.to_list()
        self.all_data = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.freq)
        self.prices = self.all_data.xs('Adj Close', axis=1).interpolate(method='time')
        self.returns = self.prices.pct_change().dropna()

    def fetch_field(self, field='Volume'):
        """Return data stored in all data. Open, Close, Volume, High, Low."""
        return self.all_data.xs(field, axis=1)


if __name__ == '__main__':
    from datetime import date
    data = DataCollector(
        ['GOOG', 'AAPL', 'META', 'AMZN'], date(2000, 1, 1), date(2023, 1, 1))
