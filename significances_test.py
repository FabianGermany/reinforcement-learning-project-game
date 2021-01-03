import numpy as np
import pandas as pd
import scipy
from scipy import stats


data_random =  [-0.051,
                -0.058,
                -0.047,
                -0.041,
                -0.043,
                -0.049,
                -0.046,
                -0.048,
                -0.037,
                -0.054,
                -0.048,
                -0.046,
                -0.044,
                -0.048,
                -0.055,
                -0.043,
                -0.045,
                -0.055,
                -0.044,
                -0.036,
                -0.046,
                -0.041,
                -0.049,
                -0.048,
                -0.051,
                -0.043,
                -0.047,
                -0.046,
                -0.047,
                -0.052,
                -0.048,
                -0.052,
                -0.045,
                -0.055,
                -0.045]


data_trained = [-0.004,
                -0.026,
                0.032,
                -0.003,
                0.061,
                -0.004,
                -0.016,
                -0.005,
                0.054,
                -0.001,
                -0.022,
                -0.017,
                0.062,
                0.086,
                0.066,
                0.055,
                -0.019,
                0.046,
                -0.020,
                0.143,
                -0.021,
                0.067,
                -0.004,
                0.066,
                0.017,
                -0.005,
                -0.024,
                -0.022,
                -0.016,
                -0.013,
                -0.091,
                -0.090,
                -0.090,
                -0.091,
                -0.091]

print(np.mean(data_random))
print(np.mean(data_trained))
print(np.std(data_random))
print(np.std(data_trained))

stat, pvalue = scipy.stats.mannwhitneyu(x=data_random,
                         y= data_trained,
                         alternative='two-sided')

print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
alpha = 0.05
if pvalue > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')