"""Script for the Week 4 assignment."""
import codecamp
from pathlib import Path
import numpy as np
import os

DATA_DIR = Path('./data')
path_load_wind =DATA_DIR / 'wind_12_ms_TI_0.1.txt'
path_ct_file = DATA_DIR / 'CT.txt'
path_param_file = DATA_DIR/ 'turbie_parameters.txt'

t = 1
y = [1, 2, 3, 4]

t_wind,u_wind = codecamp.load_wind(path_load_wind)
Ct = codecamp.calculate_ct(u_wind,path_ct_file)

turbie_dict = codecamp.load_turbie_parameters(path_param_file)
M,C,K = codecamp.get_turbie_system_matrices(path_param_file)
area = np.pi * (turbie_dict['Dr']/2)**2
homogenous_dydt = codecamp.calculate_dydt(t,y,M,C,K,rho=None,ct=None,rotor_area=None,t_wind=None,u_wind=None)
forced_dydt = codecamp.calculate_dydt(t,y,M,C,K,turbie_dict['rho'],Ct,area,t_wind,u_wind)

t_sim,u_wind_sim,xb_sim,xt_sim = codecamp.simulate_turbie(path_load_wind,path_param_file,path_ct_file)


# define the path to the folder where you want to create the file 
folder_path = './resp' 
 
# create the folder if it doesn't exist 
if not os.path.exists(folder_path): 
    os.makedirs(folder_path) 
 
# define the file name and path 
file_name = 'test_resp.txt' 
file_path = os.path.join(folder_path, file_name) 
 
# create the file 
with open(file_path, 'w') as f: 
    f.write(codecamp.save_resp(t_sim,u_wind_sim,xb_sim,xt_sim,file_path)) 

