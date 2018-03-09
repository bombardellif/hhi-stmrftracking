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
import pdb
import numpy as np
import mvutils
import math

THRESHOLD = 0.00001
test_cases_in = [
    [[0,0]],
    [[0,0],[1,2]],
    [[0,0],[2,1],[0,0]],
    [[0,0],[1/2,math.sqrt(3)/2]],
    [[0,0],[1/2,math.sqrt(3)/2],[0,0],[1/2,math.sqrt(3)/2]],
    [[0,0],[1/2,math.sqrt(3)/2],[0,0],[1/2,math.sqrt(3)/2],[1/2,math.sqrt(3)/2]],
    [[0,0],[1/2,math.sqrt(3)/2],[1,math.sqrt(3)],[1,math.sqrt(3)]],
    [[0,0],[3/2,3*math.sqrt(3)/2],[1,math.sqrt(3)]],
    [[1/2,math.sqrt(3)/2],[math.sqrt(3)/2,1/2]],
    [[0,1],[1,0]],
    [[0,1],[-1,0]],
    [[0,-1],[1,0]],
    [[0,-1],[-1,0]],
    [[0,-1],[0,1]],
    [[1,0],[-1,0]],
    [[1/2,math.sqrt(3)/2],[-1/2,math.sqrt(3)/2]],
    [[1/2,math.sqrt(3)/2],[1/2,-math.sqrt(3)/2]],
    [[-math.sqrt(2)/2,math.sqrt(2)/2],[-math.sqrt(2)/2,-math.sqrt(2)/2]],
    [[-1/2,-math.sqrt(3)/2],[1/2,-math.sqrt(3)/2]],
    [[1/2,math.sqrt(3)/2],[math.sqrt(3)/2,1/2],[math.sqrt(2)/2,math.sqrt(2)/2]],
    [[1/2,math.sqrt(3)/2],[math.sqrt(3)/2,1/2],[math.sqrt(3)/2,-1/2]],
    [[1/2,math.sqrt(3)/2],[math.sqrt(3)/2,1/2],[math.sqrt(2)/2,math.sqrt(2)/2],\
        [math.sqrt(3)/2,-1/2]],
    [[1/2,math.sqrt(3)/2],[math.sqrt(3)/2,1/2],[math.sqrt(2)/2,math.sqrt(2)/2],\
        [math.sqrt(3)/2,-1/2],[1/2,-math.sqrt(3)/2]],
    [[1/2,math.sqrt(3)/2],[math.sqrt(3)/2,1/2],[1/2,-math.sqrt(3)/2],\
        [1,0],[0,1],[-math.sqrt(3)/2,1/2],[-1,0]],
    [[1/2,-math.sqrt(3)/2],[0,-1],[-1,0]]
]
test_cases_out = [
    [0,0],
    [1/2,2/2],
    [0,0],
    [1/4,math.sqrt(3)/4],
    [1/4,math.sqrt(3)/4],
    [1/2,math.sqrt(3)/2],
    (3/2)*np.array([1/2,math.sqrt(3)/2]),
    [1,math.sqrt(3)],
    [math.sqrt(2)/2,math.sqrt(2)/2],
    [math.sqrt(2)/2,math.sqrt(2)/2],
    [-math.sqrt(2)/2,math.sqrt(2)/2],
    [math.sqrt(2)/2,-math.sqrt(2)/2],
    [math.sqrt(2)/2,math.sqrt(2)/2],  # Should actually be [-math.sqrt(2)/2,-math.sqrt(2)/2]
    [1,0],
    [0,1],
    [0,1],
    [1,0],
    [-1,0],
    [0,1], # should actually be [0,-1]
    [np.sin(np.radians(np.mean([45,30]))),np.cos(np.radians(np.mean([45,30])))],
    # This would also be a valid result: 
    #   [np.sin(np.radians(np.mean([45,60]))),np.cos(np.radians(np.mean([45,60])))]
    [math.sqrt(2)/2,math.sqrt(2)/2],
    [np.sin(np.radians(np.mean([45,30]))),np.cos(np.radians(np.mean([45,30])))],
    # This would also be a valid result: 
    #   [np.sin(np.radians(np.mean([45,60]))),np.cos(np.radians(np.mean([45,60])))]
    [math.sqrt(2)/2,math.sqrt(2)/2],
    [math.sqrt(2)/2,math.sqrt(2)/2],
    [np.sin(np.radians(np.mean([150,180]))),np.cos(np.radians(np.mean([150,180])))]
]

for inp,outp in zip(test_cases_in, test_cases_out):
    try:
        assert (np.fabs(mvutils.polar_vector_median(np.array(inp)) - outp) < THRESHOLD).all()
    except:
        pdb.set_trace()
    print(inp, '=', outp)
