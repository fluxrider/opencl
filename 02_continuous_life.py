# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

# powershell profiling: Measure-Command {start-process python 02_continuous_life.py -Wait}

import time
g0 = time.perf_counter_ns()

t0 = time.perf_counter_ns()
import PIL.Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import pyopencl as cl
print(f"#impo: {time.perf_counter_ns() - t0}")

# setup OpenCL
device = cl.get_platforms()[0].get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)

# life parameters
aliveThreshold = .2
aliveMin = .125
aliveMax = .5
birthMin = .25
birthMax = .5
# histogram paramaters
sigmaW = 2.0

# read image from file as normalized grayscale uint8
t0 = time.perf_counter_ns()
image = np.asarray(PIL.Image.open('conway_init.png').convert('L'))
out = np.empty_like(image)
print(f"#load: {time.perf_counter_ns() - t0}")

# create buffers that hold images for OpenCL (the 'device' is the gpu), and copy the input image data
grayscale_format = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
in_device = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, grayscale_format, shape=image.shape, hostbuf=image)
out_device = cl.Image(context, cl.mem_flags.WRITE_ONLY, grayscale_format, shape=out.shape)

# load and compile OpenCL program
program = cl.Program(context, open('02_continuous_life.cl').read()).build()

# execute
H = image.shape[0]
W = image.shape[1]

# pre-calculated cdf
#from scipy.stats import norm
#sigmaK = .05078125
#gauss = norm(loc=0, scale=sigmaK)
#gauss.cdf(intensity - aliveThreshold)
cdf = [4.1002868059894214e-05, 5.640724836844843e-05, 7.71601683229369e-05, 0.00010495253215914904, 0.000141950942025149, 0.00019091212798337873, 0.0002553189834422127, 0.00033954059762994715, 0.000449018463647499, 0.0005904806558555027, 0.0007721851808671094, 0.00100419288148976, 0.0012986691868776873, 0.00167021263747488, 0.0021362064644342583, 0.0027171875800412783, 0.0034372251673762213, 0.004324298695520636, 0.005410662707332165, 0.00673318323245659, 0.008333628296724809, 0.010258892881623552, 0.012561137004142466, 0.015297814519869704, 0.018531569985197262, 0.02232998162351338, 0.026765130278892338, 0.031912977326401165, 0.03785253890752262, 0.04466484957512365, 0.052431715392380125, 0.06123426457836804, 0.07115131268729906, 0.08225756872175433, 0.09462171810968817, 0.10830442765594317, 0.12335632590665702, 0.13981601932050147, 0.15770820972230698, 0.17704198127046072, 0.19780932523103623, 0.21998396796494118, 0.24352056158625357, 0.26835428778710924, 0.2944009135636684, 0.32155732340466014, 0.349702536461716, 0.37869919998576024, 0.40839553266272566, 0.4386276742493931, 0.46922238194090615, 0.5, 0.5307776180590936, 0.5613723257506065, 0.5916044673372742, 0.6213008000142397, 0.6502974635382837, 0.6784426765953395, 0.7055990864363314, 0.7316457122128908, 0.7564794384137463, 0.7800160320350584, 0.8021906747689636, 0.8229580187295393, 0.842291790277693, 0.8601839806794982, 0.8766436740933428, 0.8916955723440568, 0.9053782818903118, 0.9177424312782455, 0.928848687312701, 0.9387657354216319, 0.9475682846076198, 0.9553351504248763, 0.9621474610924773, 0.9680870226735988, 0.9732348697211076, 0.9776700183764866, 0.9814684300148028, 0.9847021854801302, 0.9874388629958575, 0.9897411071183764, 0.9916663717032752, 0.9932668167675434, 0.9945893372926679, 0.9956757013044794, 0.9965627748326238, 0.9972828124199588, 0.9978637935355658, 0.9983297873625251, 0.9987013308131223, 0.9989958071185102, 0.9992278148191329, 0.9994095193441445, 0.9995509815363525, 0.99966045940237, 0.9997446810165578, 0.9998090878720166, 0.9998580490579748, 0.9998950474678409, 0.9999228398316771, 0.9999435927516316, 0.9999589971319401, 0.9999703634695014, 0.99997870043806, 0.999984779087158, 0.9999891848110809, 0.9999923590439224, 0.9999946324215265, 0.9999962509325773, 0.999997396369176, 0.9999982021886088, 0.999998765716884, 0.9999991574632541, 0.9999994281740302, 0.9999996141331151, 0.9999997411146435, 0.999999827308238, 0.9999998854677464, 0.9999999244779085, 0.9999999504882513, 0.9999999677277916, 0.9999999790861768, 0.999999986525249, 0.9999999913684475, 0.9999999945028665, 0.9999999965193427, 0.9999999978088994, 0.9999999986286827, 0.9999999991467295, 0.9999999994721541, 0.999999999675363, 0.9999999998015012, 0.9999999998793339, 0.9999999999270744, 0.9999999999561834, 0.9999999999738264, 0.9999999999844564, 0.999999999990823, 0.9999999999946135, 0.9999999999968567, 0.9999999999981765, 0.9999999999989483, 0.9999999999993969, 0.9999999999996563, 0.9999999999998052, 0.9999999999998902, 0.9999999999999385, 0.9999999999999658, 0.999999999999981, 0.9999999999999896, 0.9999999999999943, 0.9999999999999969, 0.9999999999999983, 0.9999999999999991, 0.9999999999999996, 0.9999999999999998, 0.9999999999999999, 0.9999999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# isotropic histogram filter
# TODO put loop on GPU
t0 = time.perf_counter_ns()
map = np.empty((W, H))
for y in range(H):
  for x in range(W):
    map[x, y] = cdf[image[y, x]]
smooth = gaussian_filter(map, sigma=sigmaW)
print(f"#smoo: {time.perf_counter_ns() - t0}")

# for each pixel
# TODO put loop on GPU
t0 = time.perf_counter_ns()
for y in range(H):
  for x in range(W):
    life = image[y, x] / 255
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
print(f"#life: {time.perf_counter_ns() - t0}")

t0 = time.perf_counter_ns()
PIL.Image.fromarray(out).save('out.png')
print(f"#save: {time.perf_counter_ns() - t0}")
print(f"#tota: {time.perf_counter_ns() - g0}")

#load: 17484900
#smoo: 53050300
#life: 619943800
#save: 5753800
#tota: 1226519200