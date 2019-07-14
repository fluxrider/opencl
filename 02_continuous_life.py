# A variant of Conway Game of Life which uses continuous values.
# It uses an Isotropic Histogram Filter to gather information about the neighborhood. Filter is based on:
# Smoothed Local Histogram Filters Pixar Technical Memo 10-02 by Michael Kass and Justin Solomon

# imports
import PIL.Image
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import pyopencl as cl
import tkinter as tk
import PIL.ImageTk

# load opencl program
device = cl.get_platforms()[0].get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)
program = cl.Program(context, open('02_continuous_life.cl').read()).build()

# read image from file as grayscale uint8
image = np.asarray(PIL.Image.open('conway_init.png').convert('L'))

# init gui
root = tk.Tk()
image_tk_persistent = None
canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0])
canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=image_tk_persistent)
canvas.pack()

def heartbeat(canvas, canvas_image):
  global image
  global context
  global queue
  global program
  global device
  format_u8 = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
  format_f32 = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.FLOAT)

  # update canvas
  global image_tk_persistent
  image_tk_persistent = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image))
  canvas.itemconfig(canvas_image, image=image_tk_persistent)

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

  image = out
  canvas.after(1000, heartbeat, canvas, canvas_image)

heartbeat(canvas, canvas_image)
root.mainloop()
