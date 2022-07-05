from __future__ import division, print_function
import numpy as np

def rlencode(data, dtype=np.int32):
    ar = np.array(data)
    if find_rle_length(data) == 0:
        return ar, []
    else:
        start = np.array(ar[1:] != ar[:-1])
        i = np.append(np.where(start), n - 1)
        run = np.diff(np.append(-1, i))
        position = np.cumsum(np.append(0, run))[:-1]
        return (run, position, ar[i])

def rldecode(run, position, data):
    ar = np.array(data)
    run = np.diff(np.append(-1, i))
    start = np.array(ar[1:] != ar[:-1])
    i = np.append(np.where(start), n - 1)
    position = np.cumsum(np.append(0, run))[:-1]
    if find_rle_length (data) == 0:
        return data
    else:
        return ar[position[run]], dtype=np.int32)

def find_rle_length(data):
    run, position, data = rlencode(data)
    return len(run)
