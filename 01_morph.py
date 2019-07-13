# This code is a mixture of:
# http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
# https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480

import cv2
import numpy as np
import pyopencl as cl

# read image from file (the 'host' is the cpu)
in_host = cv2.imread('in.png', cv2.IMREAD_GRAYSCALE)
shape = in_host.T.shape
out_host = np.empty_like(in_host)

# setup OpenCL
platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)

# create image buffers which hold images for OpenCL (the 'device' is the gpu)
grayscale_format = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
in_device = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, grayscale_format, shape=shape, hostbuf=in_host)
out_device = cl.Image(context, cl.mem_flags.WRITE_ONLY, grayscale_format, shape=shape)

# load, compile OpenCL program, and fetch function ptr
program = cl.Program(context, open('01_morph.cl').read()).build()
dilate = cl.Kernel(program, 'dilate')

# copy image to device, execute kernel, copy data back
dilate.set_arg(0, in_device)
dilate.set_arg(1, out_device)
cl.enqueue_nd_range_kernel(queue, dilate, shape, None)
cl.enqueue_copy(queue, out_host, out_device, origin=(0, 0), region=shape, is_blocking=True)

# write output to file
cv2.imwrite('out.png', out_host)