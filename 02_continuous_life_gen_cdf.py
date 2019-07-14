import numpy as np
from scipy.stats import norm
sigmaK = .05078125
gauss = norm(loc=0, scale=sigmaK)
aliveThreshold = .2
for i in range(256):
  cdf = gauss.cdf(i/255 - aliveThreshold)
  print(f"{np.float32(cdf)}f, ", end = '')

print()
for i in range(256):
  cdf = gauss.cdf(i/255 - aliveThreshold)
  print(f"{np.float32(cdf)}, ", end = '')
