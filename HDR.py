#!/usr/bin/python
# OpenCV tutorial
import numpy as np
import rawpy, imageio
from skimage.exposure import rescale_intensity

raw4 = rawpy.imread('/home/simba/Downloads/Kitchen4.CR2')
raw3 = rawpy.imread('/home/simba/Downloads/Kitchen3.CR2')
raw2 = rawpy.imread('/home/simba/Downloads/Kitchen2.CR2')
raw1 = rawpy.imread('/home/simba/Downloads/Kitchen1.CR2')
raw0 = rawpy.imread('/home/simba/Downloads/Kitchen0.CR2')
rgb = raw.postprocess(no_auto_bright=True, use_auto_wb =False,gamma=None)
imageio.imwrite('/home/simba/Downloads/example.jpg', rgb)
