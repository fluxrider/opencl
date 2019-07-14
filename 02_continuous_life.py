# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

# imports
import PIL.Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import pyopencl as cl

# load opencl program
device = cl.get_platforms()[0].get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)
program = cl.Program(context, open('02_continuous_life.cl').read()).build()
format_u8 = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
format_f32 = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.FLOAT)

# read image from file as grayscale uint8 and send it to the gpu
image = np.asarray(PIL.Image.open('conway_init.png').convert('L'))

for j in range(100):
  image_gpu = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, format_u8, shape=image.shape, hostbuf=image)

  # isotropic histogram filter
  map_gpu = cl.Image(context, cl.mem_flags.WRITE_ONLY, format_f32, shape=image.shape)
  program.map_cdf(queue, image.shape, None, map_gpu, image_gpu)
  map = np.empty(image.shape, dtype='float32')
  cl.enqueue_copy(queue, map, map_gpu, origin=(0, 0), region=image.shape, is_blocking=True)
  smooth = gaussian_filter(map, sigma=2.0)

  # run game of life
  out_gpu = cl.Image(context, cl.mem_flags.WRITE_ONLY, format_u8, shape=image.shape)
  smooth_gpu = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, format_f32, shape=image.shape, hostbuf=smooth.astype('float32'))
  program.continuous_life(queue, image.shape, None, out_gpu, image_gpu, smooth_gpu)
  out = np.empty_like(image)
  cl.enqueue_copy(queue, out, out_gpu, origin=(0, 0), region=image.shape, is_blocking=True)

  # save
  PIL.Image.fromarray(out).save(f'out{j}.png')
  
  image = out