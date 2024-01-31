import numpy as np

test = ['a','b','b','c']

test = np.unique(test)

print(test)

visibility_values = [0.05,1.3,0.002]

flag = all(visibility >= 0.1 for visibility in visibility_values)

print(flag)