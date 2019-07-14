# Pre-calculate gaussian cdf() for the continuous conway game of life

import numpy as np
from scipy.stats import norm
import struct

# histogram paramaters
depth = 256
samples = 20
sigmaK = 13.0

file = open(f"out.{depth}.{samples}.{sigmaK}.cdf","wb")
data = []

s = np.arange(samples) * ((depth - 1) / (samples - 1))
gauss = norm(loc=0, scale=sigmaK)
for i in range(samples):
  for intensity in range(depth):
    data.append(gauss.cdf(intensity - s[i]))
file.write(struct.pack('d'*len(data), *data))
file.close()

# verify
#cdf = np.fromfile(f"out.{depth}.{samples}.{sigmaK}.cdf", dtype='double').reshape(samples,depth)
#for i in range(samples):
#  for intensity in range(depth):
#    if gauss.cdf(intensity - s[i]) != cdf[i][intensity]:
#      print(f"{gauss.cdf(intensity - s[i])} VS {cdf[i][intensity]}")