# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

# powershell profiling: Measure-Command {start-process python 02_continuous_life.py -Wait}

import time
g0 = time.perf_counter_ns()

# imports (note: this has the most significant load time)
t0 = time.perf_counter_ns()
import PIL.Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import pyopencl as cl
print(f"#impo: {time.perf_counter_ns() - t0}")

# histogram paramaters
sigmaW = 2.0

# setup OpenCL
t0 = time.perf_counter_ns()
device = cl.get_platforms()[0].get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)
# load and compile OpenCL program
program = cl.Program(context, open('02_continuous_life.cl').read()).build()
print(f"#prog: {time.perf_counter_ns() - t0}")

# read image from file as normalized grayscale uint8
t0 = time.perf_counter_ns()
image = np.asarray(PIL.Image.open('conway_init.png').convert('L'))
out = np.empty_like(image)
H = image.shape[0]
W = image.shape[1]
# send image to gpu
grayscale_format = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
float_format = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.FLOAT)
image_gpu = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, grayscale_format, shape=image.shape, hostbuf=image)
print(f"#load: {time.perf_counter_ns() - t0}")

# isotropic histogram filter
# TODO put loop on GPU
t0 = time.perf_counter_ns()
map = np.empty(image.shape, dtype='float32')

map_gpu = cl.Image(context, cl.mem_flags.WRITE_ONLY, float_format, shape=image.shape)
program.map_cdf(queue, image.shape, None, map_gpu, image_gpu)
cl.enqueue_copy(queue, map, map_gpu, origin=(0, 0), region=image.shape, is_blocking=True)

#cdf = [4.100286969332956e-05, 5.640724702971056e-05, 7.716016989434138e-05, 0.00010495253081899136, 0.0001419509353581816, 0.00019091213471256196, 0.00025531896972097456, 0.00033954059472307563, 0.0004490184655878693, 0.0005904806312173605, 0.0007721851579844952, 0.0010041928617283702, 0.0012986691435799003, 0.0016702126013115048, 0.0021362064871937037, 0.0027171876281499863, 0.003437225241214037, 0.004324298817664385, 0.005410662852227688, 0.006733183283358812, 0.008333628065884113, 0.010258892551064491, 0.012561136856675148, 0.015297814272344112, 0.018531570211052895, 0.02232998237013817, 0.0267651304602623, 0.03191297873854637, 0.037852540612220764, 0.04466484859585762, 0.05243171378970146, 0.06123426556587219, 0.07115131616592407, 0.08225756883621216, 0.09462171792984009, 0.10830442607402802, 0.12335632741451263, 0.139816015958786, 0.15770821273326874, 0.17704197764396667, 0.19780932366847992, 0.21998396515846252, 0.2435205578804016, 0.26835429668426514, 0.2944009006023407, 0.32155731320381165, 0.34970253705978394, 0.3786992132663727, 0.40839552879333496, 0.4386276602745056, 0.46922239661216736, 0.5, 0.530777633190155, 0.5613723397254944, 0.591604471206665, 0.6213008165359497, 0.6502974629402161, 0.678442656993866, 0.7055990695953369, 0.7316457033157349, 0.7564794421195984, 0.7800160050392151, 0.8021906614303589, 0.8229579925537109, 0.8422917723655701, 0.8601839542388916, 0.8766436576843262, 0.8916955590248108, 0.9053782820701599, 0.9177424311637878, 0.9288486838340759, 0.9387657642364502, 0.9475682973861694, 0.9553351402282715, 0.9621474742889404, 0.9680870175361633, 0.9732348918914795, 0.9776700139045715, 0.9814684391021729, 0.9847021698951721, 0.9874388575553894, 0.9897410869598389, 0.9916663765907288, 0.9932668209075928, 0.9945893287658691, 0.9956756830215454, 0.9965627789497375, 0.9972828030586243, 0.99786376953125, 0.998329758644104, 0.9987013339996338, 0.9989957809448242, 0.9992278218269348, 0.9994094967842102, 0.999550998210907, 0.9996604323387146, 0.9997446537017822, 0.9998090863227844, 0.999858021736145, 0.9998950362205505, 0.9999228119850159, 0.9999436140060425, 0.9999589920043945, 0.9999703764915466, 0.9999787211418152, 0.9999848008155823, 0.9999892115592957, 0.9999923706054688, 0.9999946355819702, 0.9999962449073792, 0.9999973773956299, 0.9999982118606567, 0.9999987483024597, 0.9999991655349731, 0.9999994039535522, 0.9999996423721313, 0.9999997615814209, 0.9999998211860657, 0.9999998807907104, 0.9999999403953552, 0.9999999403953552, 0.9999999403953552, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#for y in range(H):
#  for x in range(W):
#    map[y, x] = cdf[image[y, x]]

smooth = gaussian_filter(map, sigma=sigmaW)
print(f"#smoo: {time.perf_counter_ns() - t0}")

# run game of life
t0 = time.perf_counter_ns()
# create buffers that hold images for OpenCL, and copy the input image data and smooth matrix
out_gpu = cl.Image(context, cl.mem_flags.WRITE_ONLY, grayscale_format, shape=image.shape)
smooth_gpu = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, float_format, shape=image.shape, hostbuf=smooth.astype('float32'))
program.continuous_life(queue, image.shape, None, out_gpu, image_gpu, smooth_gpu)
# copy output back from gpu
cl.enqueue_copy(queue, out, out_gpu, origin=(0, 0), region=image.shape, is_blocking=True)
print(f"#life: {time.perf_counter_ns() - t0}")

# save
t0 = time.perf_counter_ns()
PIL.Image.fromarray(out).save('out.png')
print(f"#save: {time.perf_counter_ns() - t0}")

print(f"#tota: {time.perf_counter_ns() - g0}")
#impo: 442272900
#prog: 101194400
#load: 18757700
#smoo: 52510100
#life: 8713500
#save: 5025900
#tota: 629173700