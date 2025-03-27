# importing necessary libraries
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# importing functions
from codecamp import calculate_for_TI, plot_mean_stdv

# recording time at start of code
start_time = datetime.now()

# step 1:
# load paths for data which will be used in the
# functions called within the code below
FILE_PATH = Path(__file__)      # path to this file
FILE_DIR = FILE_PATH.parent.parent     # path to main folder 
DATA_DIR = FILE_DIR / 'data'
TI_01_DIR = FILE_DIR / 'data/wind_TI_0.1'
TI_005_DIR = FILE_DIR / 'data/wind_TI_0.05'
TI_015_DIR = FILE_DIR / 'data/wind_TI_0.15'
turbie_params = DATA_DIR / 'turbie_parameters.txt'
path_ct = DATA_DIR/'CT.txt'

# step 2:
# call calculate_for_TI which contains a loop to calculate the stdv
# and means of the balde and tower deflection for all text files
# for the associated TI value
blade_data_TI_01, tower_data_TI_01 = calculate_for_TI(
    TI_01_DIR,  path_ct, turbie_params)

blade_data_TI_005, tower_data_TI_005 = calculate_for_TI(
    TI_005_DIR, path_ct, turbie_params)

blade_data_TI_015, tower_data_TI_015 = calculate_for_TI(
    TI_015_DIR, path_ct, turbie_params)

# convert from list to numpy arrays for plotting
blade_mean_stdv_TI_01 = np.array(blade_data_TI_01)
tower_mean_stdv_TI_01 = np.array(tower_data_TI_01)

blade_mean_stdv_TI_005 = np.array(blade_data_TI_005)
tower_mean_stdv_TI_005 = np.array(tower_data_TI_005)

blade_mean_stdv_TI_015 = np.array(blade_data_TI_015)
tower_mean_stdv_TI_015 = np.array(tower_data_TI_015)


# step 3: calling plotting function to create
# subplot for blade and tower deflections for TI 0.1
plot_mean_stdv(
    blade_mean_stdv_TI_01, tower_mean_stdv_TI_01, "TI = 0.1")

plot_mean_stdv(
    blade_mean_stdv_TI_005, tower_mean_stdv_TI_005, "TI = 0.05")

plot_mean_stdv(
    blade_mean_stdv_TI_015, tower_mean_stdv_TI_015, "TI = 0.15")


# record end time of code before calling plt.show() since plt.show()
# keeps the code open until the figure is closed,
# but we are interested in if the code produces the figures in under 10 mins
end_time = datetime.now()

# printing execution time to verify running in under 10 mins
print(f"Run time: {(end_time - start_time)}")

# calling plt.show() to display the figure
plt.show()

print("The script is running empty loops to keep plots open")
input("Press Enter to exit...")  # Keeps plots open until user presses Enter

