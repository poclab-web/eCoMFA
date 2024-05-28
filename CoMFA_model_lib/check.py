# import numpy as np
# new=np.load("C:/Users/poclabws/grid_coordinates/-4.25 -4.75 -5.75 14 20 24 0.5 20240510/ptp0.50_.npy")
# old=np.load("C:/Users/poclabws/grid_coordinates/-4.25 -4.75 -5.75 14 20 24 0.5 20240510/ptp0.50.npy")
# print(new)
# print(old)

import numpy as np
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8],[9,10]])

c = np.concatenate([a,b], axis = 1) 
print(c)

#または

c = np.c_[a,b]
print(c)

#または

c = np.hstack([a,b])
print(c)

# 結果:
#  [[ 1  2  3  7  8]
#  [ 4  5  6  9 10]]

import re
ans=re.match("Gaussian.*_validation_RMSE", "Gaussian1.00_validation_RMSE")
print(ans)