# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

import PIL.Image # pip install Pillow
import numpy as np
import pyopencl as cl
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# LIFE parameters
aliveThreshold = .2
aliveMin = .125
aliveMax = .5
birthMin = .25
birthMax = .5
# smoothing historgram paramaters
depth = 256
samples = 20
sigmaK = 13.0
sigmaW = 2.0

# read image from file as grayscale
image = np.asarray(PIL.Image.open('conway_init_small.png').convert('L'))

# execute
H = image.shape[0]
W = image.shape[1]
out = np.zeros(image.shape)

# isotropic histogram filter
# compute values for s[i]
s = np.arange(samples) * ((depth - 1) / (samples - 1))
# Storage
lookup = np.zeros((samples, depth))
map = np.zeros((samples, W, H))
smooth = np.empty((samples, W, H))

for i in range(samples):
  print(i)
  # compute lookup table for each intensities
  gauss = norm(loc=0, scale=sigmaK)
  for intensity in range(depth):
    lookup[i, intensity] = gauss.cdf(intensity - s[i])
  # map each pixel of image
  for y in range(H):
    for x in range(W):
      # scale pixel intensity to depth
      intensity = int((image[y, x] / 255) * (depth - 1))
      # map with lookup
      map[i, x, y] = lookup[i, intensity]

  # smooth result
  smooth[i] = gaussian_filter(map[i], sigma=sigmaW)

  # Cache interpolators
  interpolation = [[0] * H for i in range(W)] # [W][H]
  for y in range(H):
    for x in range(W):
      interpolation[x][y] = interp1d(s, smooth[:,x,y], kind='cubic')

# for each pixel
for y in range(H):
  for x in range(W):
    # convert pixel to life force
    life = image[y, x] / 255
    # get the percentage of life in the neighborhood
    aliveNeighborhood = interpolation[x][y](aliveThreshold * 255)

    # I need to remove the life force at the current position from the neighborhood count, but I'm not sure how to compute the population percentage
    # value of one member of the neighborhood
    # I'm doing an approximation that gaussianW of 1 is approx 3x3 window, and gaussianW of 6 is approx 15x15
    if life >= aliveThreshold:
      n = 43.2 * sigmaW - 34.2
      aliveNeighborhood -= 1 / n

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
    print((x,y))
    
PIL.Image.fromarray(np.uint8(out)).save('out.png')
