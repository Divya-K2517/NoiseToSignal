import sys
import numpy as np
import matplotlib
import math

#using this file to test and learn
softmax_outputs = np.array([[0.7,0.1,0.2],
                 [0.1,0.5,0.4],
                  [0.02,0.9,0.08]])
class_targets = [0,1,1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])

