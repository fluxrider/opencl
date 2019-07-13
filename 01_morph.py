# This code is a derived mixture of:
# https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480
# http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
#
# Tested with OpenCL 1.2 CUDA 10.1.152 GeForce GTX 760 on Windows 10

import cv2 # pip install opencv-python
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

# create buffers that hold images for OpenCL (the 'device' is the gpu), and copy the input image data
grayscale_format = cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8)
in_device = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, grayscale_format, shape=shape, hostbuf=in_host)
out_device = cl.Image(context, cl.mem_flags.WRITE_ONLY, grayscale_format, shape=shape)

# load and compile OpenCL program
program = cl.Program(context, open('01_morph.cl').read()).build()

# call dilate function, copy back result, and write to file
program.dilate(queue, shape, None, in_device, out_device)
cl.enqueue_copy(queue, out_host, out_device, origin=(0, 0), region=shape, is_blocking=True)
cv2.imwrite('out.dilate.png', out_host)

# call erode function, copy back result, and write to file
program.erode(queue, shape, None, in_device, out_device)
cl.enqueue_copy(queue, out_host, out_device, origin=(0, 0), region=shape, is_blocking=True)
cv2.imwrite('out.erode.png', out_host)