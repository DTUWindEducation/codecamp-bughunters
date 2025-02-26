"""Turbie functions.
"""
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_resp(path_resp,t_start=60):
    data = np.loadtxt(path_resp,skiprows=1)
    data = data[data[:,0]>=t_start] 
    t,u,xb,xt = np.hsplit(data,4)
    return t, u, xb, xt


def load_wind(path_wind,t_start=0):
    data = np.loadtxt(path_wind,skiprows=1)
    data = data[data[:,0]>=t_start] 
    t_wind,u_wind = np.hsplit(data,2)
    return t_wind, u_wind

def load_turbie_parameters(path_turbie): 
    file_content = '' 
    with open(path_turbie, 'r') as f:
        lines = f.readlines()
        file_content = ''.join(lines)
    for line in lines:
        if 
        print(line)
    # turbie_dict = {}
    # with open(path_turbie) as f:
    #     next(f) 
    #     for line in f: 
    #         print(line)
    #         value, *key = line.split()
    #         turbie_dict[key] = [value]
    return file_content # return turbie_dict


DATA_DIR = Path('./data')
path_resp_file = DATA_DIR / 'turbie_parameters.txt'
data1= load_turbie_parameters(path_resp_file)
print(data1)