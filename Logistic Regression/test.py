import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1, 2, 58, 7, 4])
print(np.ndim(x))

x = np.array([1, 2, 58, 7, 4]).reshape(5, 1)
print(np.ndim(x))
