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
## @package evaluation
# Library for evaluating the results of the tracking with help of ground truth
# data.

import numpy as np

def compare_masks(gt, estimate):
    TP = gt & estimate
    FP = estimate & ~gt
    FN = gt & ~estimate
    return TP,FP,FN

def precision(TP, FP):
    divisor = (TP + FP)
    return TP / divisor if divisor != 0 else 0

def recall(TP, FN):
    divisor = (TP + FN)
    return TP / divisor if divisor != 0 else 0

def f_measure(P, R):
    divisor = (P + R)
    return (2*P*R) / divisor if divisor != 0 else 0

def collect_measures(gt_stream, estimate_stream):
    accum_tp, accum_fp, accum_fn = 0,0,0
    measure_per_frame = []
    first = True
    for gt,estimate in zip(gt_stream,estimate_stream):
        if first:
            # Ignore first frame from the evaluation
            first = False
        else:
            # It may happen that GT does not exist as picture file, in these cases
            # ignore the frame in the evaluation
            if gt is not None:
                TP,FP,FN = compare_masks(gt, estimate)
                TP_sum = TP.sum()
                FP_sum = FP.sum()
                FN_sum = FN.sum()

                accum_tp += TP_sum
                accum_fp += FP_sum
                accum_fn += FN_sum

                frame_precision = precision(TP_sum, FP_sum)
                frame_recall = recall(TP_sum, FN_sum)
                measure_per_frame.append(
                    (
                        frame_precision,
                        frame_recall,
                        f_measure(frame_precision, frame_recall)
                    ))

    total_precision = precision(accum_tp, accum_fp)
    total_recall = recall(accum_tp, accum_fn)
    total_f_measure = f_measure(total_precision, total_recall)
    return {
        'total': (total_precision, total_recall, total_f_measure),
        'average': np.mean(measure_per_frame, axis=0)
    }
