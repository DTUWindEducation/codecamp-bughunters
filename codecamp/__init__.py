"""Turbie functions."""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_resp(path_resp, t_start=60):
    data = np.loadtxt(str(path_resp), skiprows=1)
    data = data[data[:, 0] >= t_start]
    t, u, xb, xt = np.hsplit(data, 4)
    return t, u, xb, xt

def load_wind(path_wind, t_start=0):
    data = np.loadtxt(str(path_wind), skiprows=1)
    data = data[data[:, 0] >= t_start]
    t_wind, u_wind = np.hsplit(data, 2)
    return t_wind, u_wind


# Part 2: Plot response data
def plot_resp(t, u, xb, xt, xlim=(60, 660)):
    """
    Plots the response data with two subplots:
    - Top: Wind speed vs. Time
    - Bottom: Blade and Tower Deflections vs. Time
    """
    fig, axs = plt.subplots(2, 1, figsize=(9, 4))

    # Top subplot: Wind speed
    axs[0].plot(t, u, label="Wind Speed", color="b")
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[0].legend()
    axs[0].grid(True)

    # Bottom subplot: Blade & Tower Deflections
    axs[1].plot(t, xb, label="Blade Deflection", color="r")
    axs[1].plot(t, xt, label="Tower Deflection", color="g")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Deflection (m)")
    axs[1].legend()
    axs[1].grid(True)

    # Apply x-limits
    axs[0].set_xlim(xlim)
    axs[1].set_xlim(xlim)

    # Improve layout and show plot
    fig.tight_layout()
    plt.show()

    return fig, axs

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

