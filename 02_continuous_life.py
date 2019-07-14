# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

import time
g0 = time.perf_counter_ns()

# imports
t0 = time.perf_counter_ns()
import PIL.Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import pyopencl as cl
print(f"#impo: {time.perf_counter_ns() - t0}")

# load opencl program
t0 = time.perf_counter_ns()
device = cl.get_platforms()[0].get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)
program = cl.Program(context, open('02_continuous_life.cl').read()).build()
format_u8 = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
format_f32 = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.FLOAT)
print(f"#prog: {time.perf_counter_ns() - t0}")

# read image from file as grayscale uint8 and send it to the gpu
t0 = time.perf_counter_ns()
image = np.asarray(PIL.Image.open('conway_init.png').convert('L'))
image_gpu = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, format_u8, shape=image.shape, hostbuf=image)
print(f"#load: {time.perf_counter_ns() - t0}")

# isotropic histogram filter
t0 = time.perf_counter_ns()
map = np.empty(image.shape, dtype='float32')
map_gpu = cl.Image(context, cl.mem_flags.WRITE_ONLY, format_f32, shape=image.shape)
program.map_cdf(queue, image.shape, None, map_gpu, image_gpu)
cl.enqueue_copy(queue, map, map_gpu, origin=(0, 0), region=image.shape, is_blocking=True)
smooth = gaussian_filter(map, sigma=2.0)
print(f"#smoo: {time.perf_counter_ns() - t0}")

# run game of life
t0 = time.perf_counter_ns()
out_gpu = cl.Image(context, cl.mem_flags.WRITE_ONLY, format_u8, shape=image.shape)
smooth_gpu = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, format_f32, shape=image.shape, hostbuf=smooth.astype('float32'))
program.continuous_life(queue, image.shape, None, out_gpu, image_gpu, smooth_gpu)
out = np.empty_like(image)
cl.enqueue_copy(queue, out, out_gpu, origin=(0, 0), region=image.shape, is_blocking=True)
print(f"#life: {time.perf_counter_ns() - t0}")

# save
t0 = time.perf_counter_ns()
PIL.Image.fromarray(out).save('out.png')
print(f"#save: {time.perf_counter_ns() - t0}")

print(f"#tota: {time.perf_counter_ns() - g0}")
#impo: 444714500
#prog: 102209400
#load: 18526000
#smoo: 10351000
#life: 1379600
#save: 5049300
#tota: 582743900