"""Script for the Week 3 assignment."""
import codecamp
from pathlib import Path

path_load_resp = Path('./data')/ 'resp_12_ms_TI_0.1.txt'
t,u,xb,xt = codecamp.load_resp(path_load_resp)

path_load_wind = Path('./data')/ 'wind_12_ms_TI_0.1.txt'
codecamp.load_wind(path_load_wind)

path_load_turbie_param = Path('./data')/ 'turbie_parameters.txt'
turbie_dict = codecamp.load_turbie_parameters(path_load_turbie_param)

codecamp.plot_resp(t, u, xb, xt)

path_param_file = Path('./data')/ 'turbie_parameters.txt'
M,C,K = codecamp.get_turbie_system_matrices(path_param_file)
# TODO! Delete the line above and add your code to solve the weekly assignment.
