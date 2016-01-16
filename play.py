import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

s = pd.Series([1,3,5,np.nan,6,8, 0])



data = np.digitize(s.values, [1,2,6])

print data.__class__

print pd.Series(data) 