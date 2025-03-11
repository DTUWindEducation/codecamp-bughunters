[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NbRStOuB)
# Our CodeCamp project
# Bughunters

(Before submission) Add a brief description here. What does a
user need to know about this code?  

This code uses turbine system parameter and thrust coefficient to model turbine behaviour. The code processes wind speeds from 4m/s to 25m/s and calculates blade and tower deflections. It shows how turbines, and specifically blade and tower deflections, respond to increasing wind speed.

## Quick-start guide
- Clone the repository
- Make sure to have all the python packages installed (numpy, matplotlib, pathlib, scipy, pandas, pytest) 
- Run main.py and the script will process all wind speed files, compute deflections and generate plots 

## How the code works
The code simulates wind turbine blade and tower deflections based on wind speed variations and turbulence instensity. This code includes three major steps: loading the data, processing wind files and generating plots.


- STEP 1: Loading Wind and Turbine Data

   - The code first initializes directories and loads required data. 
       - Wind data files: stored in ./data/wnd_TI_0.1
       - Turbine Parameters: read from turbie_parameters.txt
       - CT Curve Data: Read from CT.txt
   - Initialize storage arrays(blade_data_TI_01, tower_data_TI_01)
   - Sorting Wind Data: The code reads all files in wind_TI_01, extracts wind speed values from filenames, and sort them numerically by wind speed. This ensures the for loop processes data in increasing wind speed order


- STEP 2: Processing Each Wind Speed File (For Loop)

The code loops through each wind speed data file and makes operations to calculate the blade and tower deflections. For each wind speed file in the sorted list:
   - Extract File Path and Wind Speed
   assign wind speed value from file. Store the file path for processing.
   - Load Wind Data
   Use codecamp.load.wind() to retrieve time series (t_wind) and wind speed (u_wind).
   - Compute Ct by using codecamp.calculate_ct()
   - Get Turbine System Matrices(M,C,K)
   codecamp.get_turbie_system_matrices() load mass, damping and stiffness matrices
   - Run Turbine Simulation
   codecamp.simulate_turbie() simulates blade(xb) and tower (xt) deflections
   - Compute Mean and Standard Deviation for xb and xt
   - Store results, appending them to the storage arrays


- STEP 3: Generating Plots

After processing all wind speeds, the code plots:
   - Blade Deflection vs. Wind Speed
   - Tower Deflection vs. Wind SPeed
 

## Team contributions

(Before submission) List team members. How did the different team
members contribute?  

Each team member was assigned specific tasks, but we worked collaboratively throughout the process by reviewing each other's work, providing feedback, and managing pull requests.

Kali: complete step 1 and 2 from the pseudocode in the main.py (create the code)
Tessa: complete the README file and creating the drawIO file 
Benni: complete step 3 from the pseudocode in the main.py (create the plots)
