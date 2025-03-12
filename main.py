# importing necessary libraries 
from pathlib import Path
import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt

# importing functions 
import codecamp

# recording time at start of code 
start_time = datetime.now()

# pseudocode below: 

# step 1: 
# use load_turbie_parameters to load params
DATA_DIR = Path('./data')
TI_01_DIR = Path('./data/wind_TI_0.1')
TI_005_DIR = Path('./data/wind_TI_0.05')
TI_015_DIR = Path('./data/wind_TI_0.15')
turbie_params = DATA_DIR/ 'turbie_parameters.txt' 
path_ct = DATA_DIR/'CT.txt'


# step 2: 
# call calculate_for_TI which contains a loop to calculate the stdv and means of the balde and tower deflection for all text files 
# for the associated TI value  
blade_data_TI_01,tower_data_TI_01 = codecamp.calculate_for_TI(TI_01_DIR,path_ct,turbie_params)
blade_data_TI_005,tower_data_TI_005 = codecamp.calculate_for_TI(TI_005_DIR,path_ct,turbie_params)
blade_data_TI_015,tower_data_TI_015 = codecamp.calculate_for_TI(TI_015_DIR,path_ct,turbie_params)

# convert from list to numpy arrays for plotting  
blade_mean_stdv_TI_01 = np.array(blade_data_TI_01)
tower_mean_stdv_TI_01 = np.array(tower_data_TI_01)

blade_mean_stdv_TI_005 = np.array(blade_data_TI_005)
tower_mean_stdv_TI_005 = np.array(tower_data_TI_005)

blade_mean_stdv_TI_015 = np.array(blade_data_TI_015)
tower_mean_stdv_TI_015 = np.array(tower_data_TI_015)


# step 3: calling plotting function to create subplot for blade and tower deflections for TI 0.1
codecamp.plot_mean_stdv(blade_mean_stdv_TI_01, tower_mean_stdv_TI_01,"TI = 0.1")
plt.pause(0.1)
codecamp.plot_mean_stdv(blade_mean_stdv_TI_005, tower_mean_stdv_TI_005,"TI = 0.05")
plt.pause(0.1)
codecamp.plot_mean_stdv(blade_mean_stdv_TI_015, tower_mean_stdv_TI_015,"TI = 0.15")
plt.pause(0.1)

# record end time of code before calling plt.show() since plt.show() keeps the code open until the figure is closed, 
# but we are interested in if the code produces the figures in under 10 mins 
end_time = datetime.now()

# printing execution time to verify running in under 10 mins 
print('Run time: {}'.format(end_time - start_time))

# calling plt.show() to display the figure
plt.show()


input("Press Enter to exit, but the figures will remain open...")





