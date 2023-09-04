"""
Created on Sat Sep  2 8:26:52 2023

@author: ryanm
"""

import pandas as pd
from datetime import date

from Project1.scripts.style_analysis import 


if __name__ == '__main__':
    benchmark = pd.Series({'IVV': 1.0})
    results = main(benchmark, 90)
    results.plot(kind='area')
