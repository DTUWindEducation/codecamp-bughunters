# Here is the .py file which will be used / submitted for the project (Due March 12 at midnight)

# given 
    # Turbie parameters and CT curve.
    # Turbulent wind time series for 22 different mean wind speeds (at 3 different turbulence intensities).

# An error-free main.py script that calculates and plots the mean and standard deviation of the blade and tower deflections versus wind speed for TI = 0.1.
    # The design/format of the plot is up to you.
    # The design of the main script is also up to you. You can add more functions to __init__.py if you need.
    # Extra credit: make your main.py such that it can process/visualize results from all 3 TIs.


# pseudocode below: 

# step 1: 
# use load_turbie_parameters to load params 
# initalize two empty arrays (blade_mean_stdv and tower_mean_stdv) to be used to store standard deviations and means for each wind speed data set 

# step 2: 
# create a loop which, for each wind speed file for TI = 0.1, does the following: 
#   use load_wind to load the wind data 
#   calculate Ct from CT curve and wind time series data 

#   call get_turbie_system_matricies 
#   call simulate_turbie to get time, wind, and deflections 
#   calculate the mean and standard deviation of the blade
#   calculate the mean and stadard deviation of the tower 
#   append tower_mean_stdv and blade_mean_stdv with mean and stdv values
# end loop 

# step 3:
# create two figures (see reference link for method for plotting): 
#   one which plots the mean's and stdv's for the blade 
#   one which plots the means's and stdv's for the tower 
#       reference link (https://stackoverflow.com/questions/22481854/plot-mean-and-standard-deviation)
#   the figures should have the wind speed on the x axis (ex: 1, 2, 3, 4,.... [m/s])


