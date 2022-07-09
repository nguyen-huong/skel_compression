from __future__ import division, print_function
import numpy as np

#end of block, detect last non-0 values, take last value and decode it
def rlencode(data, dtype=np.int32):
    ar = np.array(data)
    if len(data) == 0:
        return ar, []
    else:
        # run = np.diff(np.append(1, ar))
        start = np.array(ar[1:] != ar[:-1])
        i = np.append(np.where(start), len(ar) - 1)
        run = np.diff(np.append(-1, i))
        position = np.cumsum(np.append(0, run))[:-1]
        # return (run, position, ar[i])
        return run
    #append it to the coefficient error
    #when decode, append the last value to the end of the array ((window_length*no_joints)-run)


def rldecode(run, position, data):
    ar = np.array(data)
    run = np.diff(np.append(-1, i))
    start = np.array(ar[1:] != ar[:-1])
    i = np.append(np.where(start), len(ar) - 1)
    position = np.cumsum(np.append(0, run))[:-1]
    if len(data) == 0:
        return data
    else:
        return ar[position[run]]

# def find_rle_length(data):
#     run, position, data = rlencode(data)
#     return len(run)