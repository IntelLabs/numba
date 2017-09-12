#! /usr/bin/env python

#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function

import sys
import time
import os

import numpy as np

from numba import stencil
from PIL import Image

@stencil()
def xsten(a):
    return ((a[-1,-1] * -1.0) + (a[-1,0] * -2.0) + (a[-1,1] * -1.0) + a[1,-1] + (a[1,0] * 2.0) + a[1,1]) / 12.0

@stencil()
def ysten(a):
    return ((a[-1,-1] * -1.0) + (a[0,-1] * -2.0) + (a[1,-1] * -1.0) + a[-1,1] + (a[0,1] * 2.0) + a[1,1]) / 12.0

@stencil()
def harris_common(a):
    return (a[-1,-1] + a[-1,0] + a[-1,1] + a[0,-1] + a[0,0] + a[0,1] + a[1,-1] + a[1,0] + a[1,1])

def harris(Iin):
    Ix = xsten(Iin)
    Iy = ysten(Iin)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Sxx = harris_common(Ixx)
    Syy = harris_common(Iyy)
    Sxy = harris_common(Ixy)
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    harris = det - (0.04 * trace * trace)
    return harris

def main (*args):
    iterations = 1
    #input_file = "chessboard.jpg" 
    
    if len(args) > 0:
        input_file = args[0]

    parts = os.path.splitext(input_file)
    new_file_name = parts[0] + "-corners" + parts[1]

    input_img = Image.open(input_file)
    input_arr = np.array(input_img)
    #output_arr = np.zeros_like(input_arr)
    
    tstart = time.time()
    output_arr = harris(input_arr).astype(np.uint8)
    htime = time.time() - tstart
    print("SELFTIMED ", htime)

    new_img = Image.fromarray(output_arr, mode=input_img.mode)
    new_img.format = input_img.format
    new_img.save(new_file_name)

if __name__ == "__main__":
    main(*sys.argv[1:])