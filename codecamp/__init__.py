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
