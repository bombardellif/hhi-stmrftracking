"""
  Copyright:
  2016 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
  The copyright of this software source code is the property of HHI.
  This software may be used and/or copied only with the written permission
  of HHI and in accordance with the terms and conditions stipulated
  in the agreement/contract under which the software has been supplied.
  The software distributed under this license is distributed on an "AS IS" basis,
  WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
"""
## @package ioutils
# Library of utility functions for input and output

import numpy as np
import scipy.misc
import os
import os.path
import re

MV_MAT_TYPE = np.int16

## Read the file as a **binary** picture (1-bit palette) describing the mask of
# an object, and return this mask as an HxW-ndarray of type bool, where H is
# the height and W the width of the picture.
#
# @type filename    string
# @param filename   Path to picture file in the file system
# @return   HxW-ndarray of type bool, where H is the height and W the width of
#           the frame
def read_object_mask(filename):
    return scipy.misc.imread(filename, mode='L').astype(np.bool, copy=False)

## Parse the text file in the standard form outputted by mpegflow, which
# comprises a sequence of frames and their respective motion vectors, and
# return an iterator where each element represents a frame and each element is
# a two-dimensional array. This array has shape (N,4), where N is the number of
# motion vectors in the frame. Each line of this array is of the kind
# [x,y,dx,dy], where x,y are the coordinates, and dx,dy are the values of a
# motion vector.
#
# @type filename    string
# @param filename   Path to mpegflow's output text file in the file system
# @return   Iterator of Nx4-ndarray, where N is the number of motion vectors in
#           the respective frame
def read_mv_stream(filename):
    frame = []
    is_p_frame = False
    first = True
    with open(filename) as txtFile:
        for line in txtFile:
            # If line starts with # then it is a start of a new frame
            if line[0] == '#':
                # If this is not the first frame, add the last one to the output
                if first:
                    first = False
                else:
                    result = np.array(frame, dtype=MV_MAT_TYPE)
                    if result.shape[0] > 0:
                        result[:,2:] *= -1
                    yield result
                # Start a new frame buffer
                frame = []
                is_p_frame = 'pict_type=P' in line
            else:
                # Add motion vector to frame buffer iff current frame is P-frame
                if is_p_frame:
                    frame.append(np.array(line.split()))
        # Got to the EOF, yield the last read frame
        if not first:
            result = np.array(frame, dtype=MV_MAT_TYPE)
            if result.shape[0] > 0:
                result[:,2:] *= -1
            yield result

## Read the files with the given extension in the given folder as binary images
# and return an iterator on the arrays. Result is ordered according to the file
# names, which should be composed of a number followed by the extension, e.g.
# 0.png, 1.png, etc.
#
# @type foldername  str
# @param foldername Path to folder containing the ground truth images. Must end
#                   with slash (/)
# @type extension   str
# @param extension  Image file's extension. Must begin with dot (.)
# @return Iterator of HxW-ndarray, where H is the height and W the width of the
#           frame. It may contain `None` values.
def read_gt_stream(foldername, extension='.bmp'):
    frame_num_regexp = re.compile('(\d+)\\' + extension)
    def frame_num(elem):
        frame_num_match = frame_num_regexp.search(elem)
        return int(frame_num_match.groups(0)[0]) if frame_num_match else 0

    pics = [os.path.join(foldername, filename) for filename \
        in os.listdir(foldername) if filename.endswith(extension)]
    pics_dict = {frame_num(pic): pic for pic in pics}

    frame_num_range = range(min(pics_dict), max(pics_dict) + 1)
    return (read_object_mask(pics_dict[i]) if i in pics_dict else None \
        for i in frame_num_range)

def write_mv_txt(frame, filename):
    f = open(filename, 'w')
    if frame.size:
        # Invert the vector components, invert the signal and multiply by 4
        flat_str = str(frame[:,:,::-1].flatten() * 4)
        # Write in the file without array brackets and line break
        f.write(flat_str[1:-1].replace("\n", ''))
    f.close()

def read_gmp(filename):
    res = None
    with open(filename, 'r') as txtFile:
        list_line = (line.split() for line in txtFile)
        res = [np.array((l[1], l[2], l[0], l[4], l[5], l[3]), dtype=np.float)\
              for l in list_line]
    return res

def read_mv_stream_from_decoder(tracking_id):
    from decoder import decoder
    keep_reading = True
    while keep_reading:
        keep_reading,array = decoder.get_next_frame(tracking_id)
        yield array
