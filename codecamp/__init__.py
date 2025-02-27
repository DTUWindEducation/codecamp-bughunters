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


def load_turbie_parameters(path_parameters):
    data= np.loadtxt(path_parameters, dtype=str, skiprows=1) #load data as string (both numerical values and strings)
    values, keys = np.hsplit(data, [1]) #values get first column, keys get all remaining columns
    values = data[:,0].astype(float) #convert first column to float (values)
    clean_keys= []

    # BELOW: Sorting through each list in the key list, then selecting the value in index position 1 
    # (which associates with the variable name) and assigning only this str to the cleaned_key list
    # so that the cleaned list only contains the associated variable name for each value of the data
    for k in keys: 
        clean_keys.append(k[1])

    turbie_dict = dict(zip(clean_keys, values))
    return turbie_dict



def get_turbie_system_matrices(path_parameters):
    params= load_turbie_parameters(path_parameters)

    # Define mass values
    m1 = 3 * params["mb"]  # Mass of 3 blades
    m2 = params["mn"] + params["mt"] + params["mh"]      # Mass at tower + nacelle + hub
        # Define damping coefficients
    c1 = params["c1"]
    c2 = params["c2"]

    # Define stiffness coefficients
    k1 = params["k1"]
    k2 = params["k2"]

    # Mass matrix M (2x2)
    M = np.array([
        [m1, 0],
        [0, m2]
    ])

    # Damping matrix C (2x2)
    C = np.array([
        [c1, -c1],
        [-c1, c1 + c2]
    ])

    # Stiffness matrix K (2x2)
    K = np.array([
        [k1, -k1],
        [-k1, k1 + k2]
    ])

    return M, C, K

