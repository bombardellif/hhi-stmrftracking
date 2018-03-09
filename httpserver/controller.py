import time
import numpy as np

from decoder import decoder
from hhi_stmrftracking import ioutils
from hhi_stmrftracking import imgutils
from hhi_stmrftracking import tracking
from hhi_stmrftracking.mvutils import SMALLEST_BLOCK_SIZE

results = {}
id_counter = 0

def init():
    decoder.init()

def _convert_bbox_to_ours(bbox):
    left = bbox[0] // SMALLEST_BLOCK_SIZE
    top = bbox[1] // SMALLEST_BLOCK_SIZE
    right = bbox[2] // SMALLEST_BLOCK_SIZE
    bottom = bbox[3] // SMALLEST_BLOCK_SIZE
    return (top,bottom,left,right)

def _convert_bbox_to_theirs(bbox):
    top = int(bbox[0] * SMALLEST_BLOCK_SIZE)
    bottom = int(bbox[1] * SMALLEST_BLOCK_SIZE)
    left = int(bbox[2] * SMALLEST_BLOCK_SIZE)
    right = int(bbox[3] * SMALLEST_BLOCK_SIZE)
    return "%d,%d,%d,%d" % (left,top,right,bottom)

def start(query):
    global id_counter
    trackid = id_counter
    # Get video stream from video source
    success = decoder.read_videostream(trackid, query['furl'])
    if not success:
        return {
            'success': False,
            'message': 'Failed to start tracking process.'
        }

    # Determine object's mask
    rows = query['shape'][0] // SMALLEST_BLOCK_SIZE
    cols = query['shape'][1] // SMALLEST_BLOCK_SIZE
    mask = np.zeros((rows,cols), dtype=np.bool)
    bbox = _convert_bbox_to_ours(query['bbox'])
    top,bottom,left,right = bbox
    mask[top:bottom, left:right] = True

    # Create motion vector stream and result stream
    mv_stream = ioutils.read_mv_stream_from_decoder(trackid)
    result = tracking.track(mask, mv_stream)
    results[trackid] = {
        'queue': result,
        'state': {
            'id': trackid,
            'furl': query['furl'],
            'shape': query['shape'],
            'timestamp': query['timestamp'],
            'bbox': _convert_bbox_to_theirs(bbox)
        }}
    id_counter += 1
    return results[trackid]['state']

def _terminate_tracking(trackid):
    decoder.stop(trackid)
    decoder.destroy(trackid)
    del results[trackid]

def stop(query):
    trackid = query['id']
    if trackid in results:
        _terminate_tracking(trackid)
        return {
            'id': trackid,
            'success': True,
            'message': 'Tracking process stopped.'
        }
    else:
        return {
            'id': trackid,
            'success': False,
            'message': 'Tracking process not found.'
        }

def clear(query):
    global results
    for trackid in results.keys():
        decoder.stop(trackid)
        decoder.destroy(trackid)
    results = {}
    return None

def _update_tracking(tracking, bbox):
    # update global state of this tracking process (timestamp and bbox)
    new_status = tracking['state']
    new_status['timestamp'] = round(time.time())
    if bbox is None:
        # End of tracking, stop this task
        _terminate_tracking(new_status['id'])
        new_status['bbox'] = None
    else:
        new_status['bbox'] = _convert_bbox_to_theirs(bbox)
    return new_status

def _get_update_tracking(tracking):
    # Get the tracking result of next frame
    try:
        mask = next(tracking['queue'])
        bbox = imgutils.bounding_box(mask)
    except:
        bbox = None
    return _update_tracking(tracking, bbox)

def get(query):
    trackid = query['id']
    if trackid in results:
        new_status = _get_update_tracking(results[trackid])
        response = new_status.copy()
        response['success'] = True
        return response
    else:
        return {
            'id': trackid,
            'success': False,
            'message': 'Tracking process not found.'
        }

def getlist(query):
    response = []
    for tracking in list(results.values()):
        response.append(_get_update_tracking(tracking))
    return response
