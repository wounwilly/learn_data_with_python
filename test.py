import sys
import os
import numpy as np

tab = [-10, 30, 4, 5, 10, -20]
x = np.quantile(tab, 0.25)
print(x)