import sklearn.model_selection

import ms_svm as mss
import numpy as np

svm = mss.Svm(mss.RbfKernel(.1))

x = np.array([[1],
              [1.9],
              [2.1],
              [3]])

y = np.array([1, -1, -1, 1])

svm.fit(x, y)

print(svm(x))