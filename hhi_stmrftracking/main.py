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
## Main Module

import numpy as np
import scipy.misc
import argparse
import itertools

from hhi_stmrftracking import ioutils
from hhi_stmrftracking import imgutils
from hhi_stmrftracking import mvutils
from hhi_stmrftracking import tracking
from hhi_stmrftracking import evaluation
from decoder import decoder

## Just for Evaluation/Parameter selection
set_params = tracking.set_params

# Predefined Constants
BLOCK_SIZE = mvutils.SMALLEST_BLOCK_SIZE
# BLOCK_THRESHOLD = BLOCK_SIZE**2 // 2
BLOCK_THRESHOLD = BLOCK_SIZE // 2
GT_EXTENSION = '.png'
# GT_EXTENSION = '.bmp'
GT_BLOCKIFIED = ['gt-coastguard']

def blockify_mask(mask):
    return imgutils.discretize_bw(
        imgutils.blockify(mask, BLOCK_SIZE),
        BLOCK_THRESHOLD)

def evaluate(mask_list, eval_folder, mask_pixel_shape):
    gt_stream = ioutils.read_gt_stream(eval_folder, GT_EXTENSION)
    are_gts_pixel_size = eval_folder not in GT_BLOCKIFIED

    # Create one iterator for GTs in blocks and one in pixels
    iterator_gts, blockified_gts = zip(*(
        (gt, (blockify_mask(gt) if are_gts_pixel_size else gt)\
             if gt is not None else None) \
        for gt in gt_stream))
    eval_block_domain = evaluation.collect_measures(blockified_gts, mask_list)

    # Create one iterator with results scaled up to pixel size
    if are_gts_pixel_size:
        pixels_masks = (
            scipy.misc.imresize(
                m,
                (mask_pixel_shape[0], mask_pixel_shape[1]),
                interp='nearest')\
            .astype(np.bool, copy=False) \
            for m in mask_list)
    else:
        pixels_masks = mask_list
    eval_pixel_domain = evaluation.collect_measures(iterator_gts, pixels_masks)

    eval_block_domain['pixel_total'] = eval_pixel_domain['total']
    eval_block_domain['pixel_average'] = eval_pixel_domain['average']
    return eval_block_domain

def display_masks(m0, masks, display_file, display_skip, callback):
    # Import these modules only if this function is called
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import cv2

    fig = plt.figure()
    plt.axis('off')
    if display_file != '':
        cap = cv2.VideoCapture(display_file)
        # Skip the first `display_skip` frames
        for _ in range(0, display_skip):
            ret, frame = cap.read()
        _, frame1 = cap.read()
        rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        im = plt.imshow(rgb)
        def update(enum_mask):
            i, m = enum_mask
            # Read Frame
            frame = frame1
            if i != 0:
                ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Align images and compare them
            pixel_mask = cv2.resize(
                m.view(np.uint8),
                (frame.shape[1],frame.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            # Combine mask in the blue channel
            rgb[pixel_mask.view(np.bool), 2] = 255
            rgb[pixel_mask.view(np.bool), :2] //= 2
            im.set_array(rgb)
            # Call the callback function if it evaluates to True
            callback and callback(i, m)
            return (im,)
    else:
        im = plt.imshow(m0, cmap='gray')
        def update(enum_mask):
            i, m = enum_mask
            im.set_array(m)
            # Call the callback function if it evaluates to True
            callback and callback(i, m)
            return (im,)

    anim = animation.FuncAnimation(fig,
        update,
        frames=enumerate(masks),
        interval=10,
        save_count=1000)
    plt.show()

def display_evaluation(m0, masks, display_file, display_skip, callback,
                       eval_folder):
    # Import these modules only if this function is called
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import cv2

    gt_stream = ioutils.read_gt_stream(eval_folder, GT_EXTENSION)

    fig = plt.figure()
    plt.axis('off')
    cap = cv2.VideoCapture(display_file)
    # Skip the first `display_skip` frames
    for _ in range(0, display_skip):
        ret, frame = cap.read()
    _, frame1 = cap.read()
    rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    im = plt.imshow(rgb)
    def update(enum_mask):
        i, (gt, m) = enum_mask
        # Read Frame
        frame = frame1
        if i != 0:
            ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Align images and compare them
        pixel_mask = cv2.resize(
            m.view(np.uint8),
            (frame.shape[1],frame.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        gt_resized = cv2.resize(
            gt.view(np.uint8),
            (frame.shape[1],frame.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        TP,FP,FN = evaluation.compare_masks(gt_resized, pixel_mask)
        # Combine TP in green channel, FP in blue, FN in red
        rgb[FN.view(np.bool), 0] = 255
        rgb[FN.view(np.bool), 1:] //= 2
        rgb[TP.view(np.bool), 1] = 255
        rgb[TP.view(np.bool), 0] //= 2;rgb[TP.view(np.bool), 2] //= 2;
        rgb[FP.view(np.bool), 2] = 255
        rgb[FP.view(np.bool), :2] //= 2
        im.set_array(rgb)
        # Call the callback function if it evaluates to True
        callback and callback(i, m)
        return (im,)

    anim = animation.FuncAnimation(fig,
        update,
        frames=enumerate(zip(gt_stream,masks)),
        interval=10,
        save_count=1000)
    plt.show()
    # anim.save('%s.eval.mp4' % display_file)

def output_mask(i, mask, folder):
    scipy.misc.imsave('data/out/%s/%d.png' % (folder, i),
        mask.astype(np.bool, copy=False))

def run_tracking(vector_stream, mask_file, output_folder=None, eval_folder=None,
                display_file=None, skip=0, num_frames=None):
    object_mask = ioutils.read_object_mask(mask_file)
    # Ignore also the first frame since it corresponds to the initial mask
    decoder.init()
    status = decoder.read_videostream(0, vector_stream)
    if not status:
        raise Exception('Error reading video stream')
    raw_mv_stream = ioutils.read_mv_stream_from_decoder(0)
    mv_stream = itertools.islice(
        raw_mv_stream,
        skip + 1,
        num_frames)
    return_value = None

    # Adjust the mask in 4x4 block
    mask0 = blockify_mask(object_mask) \
        if eval_folder not in GT_BLOCKIFIED else object_mask

    # Execute the tracking
    masks = tracking.track(mask0, mv_stream)

    is_eval = eval_folder is not None
    is_display = display_file is not None
    is_output = output_folder is not None
    # Evaluate results
    if is_eval and not is_display:
        mask_list = list(masks)
        return_value = evaluate(mask_list, eval_folder, object_mask.shape)
        masks = mask_list

    # Output Masks
    if is_display:
        output_callback = lambda i, m: output_mask(i, m, output_folder) \
            if is_output else None
        # For evaluation display
        if is_eval:
            display_evaluation(mask0, masks, display_file, skip, output_callback,
                               eval_folder)
        else:
            display_masks(mask0, masks, display_file, skip, output_callback)
    elif is_output:
        for i, m in enumerate(masks):
            output_mask(i, m, output_folder)

    decoder.stop(0)
    decoder.destroy(0)
    return return_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HHI ST-MRF Tracking.')
    parser.add_argument(
        'vector_stream',
        help='File path for the motion vector stream.')
    parser.add_argument(
        'mask_file',
        help='File path for the mask of the first frame.')
    parser.add_argument(
        '-o',
        '--output',
        help='Save the tracking results as images inside `folder`.',
        metavar='folder')
    parser.add_argument(
        '-e',
        '--evaluation',
        help='Perform the evaluation of results with ground truth images located in `folder`.',
        metavar='folder')
    parser.add_argument(
        '-d',
        '--display',
        help='Display the tracking result overlaid on the original video.',
        metavar='video_file')
    parser.add_argument(
        '-s',
        '--skip',
        help='Number of frames to skip at the beginning of the video (default 0).',
        metavar='skip',
        type=int,
        default=0)
    args = parser.parse_args()

    print(run_tracking(
        args.vector_stream,
        args.mask_file,
        output_folder=args.output,
        eval_folder=args.evaluation,
        display_file=args.display,
        skip=args.skip))
