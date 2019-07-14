# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

import PIL.Image # pip install Pillow
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import time
#import pyopencl as cl # TMP this is the CPU version, which is pretty slow

# life parameters
aliveThreshold = .2
aliveMin = .125
aliveMax = .5
birthMin = .25
birthMax = .5
# histogram paramaters
sigmaK = 13.0 / 256
sigmaW = 2.0

# read image from file as normalized grayscale
image = np.asarray(PIL.Image.open('conway_init.png').convert('L')) / 255

# execute
H = image.shape[0]
W = image.shape[1]
out = np.empty(image.shape)

# isotropic histogram filter
# TODO put loop on GPU
gauss = norm(loc=0, scale=sigmaK)
cdf_cache = {}
map = np.empty((W, H))
for y in range(H):
  for x in range(W):
    # scale pixel intensity to depth
    intensity = image[y, x]
    if intensity not in cdf_cache:
      cdf_cache[intensity] = gauss.cdf(intensity - aliveThreshold)
    map[x, y] = cdf_cache[intensity]
# smooth result
smooth = gaussian_filter(map, sigma=sigmaW)

# for each pixel
# TODO put loop on GPU
for y in range(H):
  print(f"row {y}")
  for x in range(W):
    # convert pixel to life force
    life = image[y, x]
    # get the percentage of life in the neighborhood
    aliveNeighborhood = smooth[x,y]

    # I need to remove the life force at the current position from the neighborhood count, but I'm not sure how to compute the population percentage
    # value of one member of the neighborhood
    # I'm doing an approximation that gaussianW of 1 is approx 3x3 window, and gaussianW of 6 is approx 15x15
    if life >= aliveThreshold:
      aliveNeighborhood -= 1 / (43.2 * sigmaW - 34.2)

    # if alive
    if life >= aliveThreshold:
      # stay alive and adjust life force if neighborhood alive population is within 12.5% to 50%, with life force being at its peak at 31.25%
      half = (aliveMax + aliveMin) / 2
      delta = half - aliveMin
      if aliveNeighborhood < aliveMin or aliveNeighborhood > aliveMax: life = 0
      else: life = 1 - (abs(aliveNeighborhood - half) / delta)
    # if almost dead or dead
    else:
      # come to life if neighbohood alive population is withing 25% to 50%, with life force being at its peak at 37.5%
      half = (birthMax + birthMin) / 2
      delta = half - birthMin
      if aliveNeighborhood < birthMin or aliveNeighborhood > birthMax: life = 0
      else: life = 1 - (abs(aliveNeighborhood - half) / delta)
    
    # convert life to rgb
    out[y, x] = max(0, min(255, life * 255))

PIL.Image.fromarray(np.uint8(out)).save('out.png')
