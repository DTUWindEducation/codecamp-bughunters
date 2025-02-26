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
