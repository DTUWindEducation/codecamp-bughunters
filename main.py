# Here is the .py file which will be used / submitted for the project (Due March 12 at midnight)

# given 
    # Turbie parameters and CT curve.
    # Turbulent wind time series for 22 different mean wind speeds (at 3 different turbulence intensities).

# An error-free main.py script that calculates and plots the mean and standard deviation of the blade and tower deflections versus wind speed for TI = 0.1.
    # The design/format of the plot is up to you.
    # The design of the main script is also up to you. You can add more functions to __init__.py if you need.
    # Extra credit: make your main.py such that it can process/visualize results from all 3 TIs.


# importing necessary libraries 
from pathlib import Path
import numpy as np
import os 
from datetime import datetime

# importing functions 
import codecamp

# recording time at start of code 
start_time = datetime.now()

# pseudocode below: 

# step 1: 
# use load_turbie_parameters to load params
DATA_DIR = Path('./data')
TI_1_DIR = Path('./data/wind_TI_0.1')
wind_directory = os.listdir(TI_1_DIR)
turbie_params = DATA_DIR/ 'turbie_parameters.txt' 
path_ct = DATA_DIR/'CT.txt'

# initalize two empty arrays (blade_mean_stdv and tower_mean_stdv) to be used to store standard deviations and means for each wind speed data set 
blade_data_TI_01 = []
tower_data_TI_01 = []

# Get all .txt files in the folder
files = [f for f in os.listdir(TI_1_DIR) if f.endswith('.txt')]

# Sort the files by extracting the wind speed value (which is the second element when splitting by '_')
files.sort(key=lambda x: int(x.split('_')[1]))

# step 2: 
# create a loop which, for each wind speed file for TI = 0.1, does the following: 
for file_name in files: 
    # looping through the file names to set the path of the wind data for each data file 
    path_wind = TI_1_DIR/(str(file_name))

    # extracting associated wind speed from the file name to correlate the mean and stdv to the associated wind speed 
    associated_spd = int(file_name.split('_')[1])

    #   use load_wind to load the wind data 
    t_wind, u_wind = codecamp.load_wind(path_wind)

    #  calculate Ct from CT curve and wind time series data 
    ct = codecamp.calculate_ct(u_wind,path_ct)

    #  call get_turbie_system_matricies 
    M, C, K = codecamp.get_turbie_system_matrices(turbie_params)

    #  call simulate_turbie to get time, wind, and deflections 
    t, wind, xb, xt = codecamp.simulate_turbie(path_wind, turbie_params,path_ct)

    #  calculate the mean and standard deviation of the blade
    mean_blade = np.mean(xb)
    std_blade = np.std(xb)

    #  calculate the mean and stadard deviation of the tower
    mean_tower = np.mean(xt)
    std_tower = np.std(xt)

    #  append tower_mean_stdv and blade_mean_stdv with mean and stdv values
    blade_data_TI_01.append([associated_spd,mean_blade,std_blade])
    tower_data_TI_01.append([associated_spd,mean_tower, std_tower])
# end loop 

blade_mean_stdv_TI_01 = np.array(blade_data_TI_01)
tower_mean_stdv_TI_01 = np.array(tower_data_TI_01)


# step 3:
# create two figures (see reference link for method for plotting): 
#   one which plots the mean's and stdv's for the blade 
#   one which plots the means's and stdv's for the tower 
#       reference link (https://stackoverflow.com/questions/22481854/plot-mean-and-standard-deviation)
#   the figures should have the wind speed on the x axis (ex: 1, 2, 3, 4,.... [m/s])

# recording time at end of code
end_time = datetime.now()

# printing execution time to verify running in under 10 mins 
print('Run time: {}'.format(end_time - start_time))


