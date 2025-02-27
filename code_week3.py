"""Script for the Week 3 assignment."""
import codecamp
from pathlib import Path

path_load_resp = Path('./data')/ 'resp_12_ms_TI_0.1.txt'
codecamp.load_resp(path_load_resp)

path_load_wind = Path('./data')/ 'wind_12_ms_TI_0.1.txt'
codecamp.load_wind(path_load_wind)

path_load_turbie_param = Path('./data')/ 'turbie_parameters.txt'
codecamp.load_turbie_parameters(path_load_turbie_param)
# TODO! Delete the line above and add your code to solve the weekly assignment.
