"""Turbie functions."""
import os
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def load_resp(path_resp, t_start=60):
    """
    Load turbine response data from a text file.

    Parameters:
        path_resp (str): Path to the response data file.
        t_start (float): Starting time in seconds to include in the data.

    Returns:
        tuple: Arrays for time (t), wind speed (u), blade deflection (xb),
        and tower deflection (xt).
    """
    # load data from text file, skipping first row since it is header info
    data = np.loadtxt(path_resp, skiprows=1)
    # selecting the data according to specified start time
    data = data[data[:, 0] >= t_start]
    # returning variables
    t, u, xb, xt = data.T
    return t, u, xb, xt


def load_wind(path_wind, t_start=0):
    """
    Load wind speed data from a text file.

    Parameters:
        path_wind (str): Path to the wind data file.
        t_start (float): Starting time in seconds to include in the data.

    Returns:
        tuple: Arrays for time (t_wind) and wind speed (u_wind).
    """
    # load data from text file, skipping first row since it is header info
    data = np.loadtxt(path_wind, skiprows=1)
    # selecting the data according to specified start time
    data = data[data[:, 0] >= t_start]
    # returning variables
    t_wind, u_wind = data.T
    return t_wind, u_wind


# Part 3: Plot response data
def plot_resp(t, u, xb, xt, xlim=(60, 660)):
    """
    Plot response data for wind speed, blade deflection, and tower deflection.

    Parameters:
        t (array): Time values.
        u (array): Wind speed data.
        xb (array): Blade deflection data.
        xt (array): Tower deflection data.
        xlim (tuple): Limits for x-axis on the plots.

    Returns:
        tuple: (Figure object, Axes object tuple)
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
    axs = tuple(axs)

    return fig, axs


def load_turbie_parameters(path_parameters):
    """
    Load turbine parameters from a text file into a dictionary.

    Parameters:
        path_parameters (str): Path to the parameter file.

    Returns:
        dict: Dictionary mapping parameter names (str) to
        their corresponding float values.
    """
    # load data as string (both numerical values and strings)
    data = np.loadtxt(path_parameters, dtype=str, skiprows=1)
    # values get first column, keys get all remaining columns
    values, keys = np.hsplit(data, [1])
    # convert first column to float (values)
    values = data[:, 0].astype(float)
    clean_keys = []

    # BELOW: Sorting through each list in the key list, then
    # selecting the value in index position 1
    # (which associates with the variable name) and assigning
    # only this str to the cleaned_key list
    # so that the cleaned list only contains the associated
    # variable name for each value of the data
    for k in keys:
        clean_keys.append(k[1])

    turbie_dict = dict(zip(clean_keys, values))
    return turbie_dict


def get_turbie_system_matrices(path_parameters):

    """ This function returns the mass, damping and stifness
    matrices for the turbine system
    given the path to the parameters file.

    Parameters:
    path_parameters (str): Path to parameters file

    Returns:
    np.ndarray: Mass matrix (M), damping matrix (C) and stifness matrix (K)
    """
    params = load_turbie_parameters(path_parameters)

    # Define mass values
    # Mass of 3 blades
    m1 = 3 * params["mb"]
    # Mass at tower + nacelle + hub
    m2 = params["mn"] + params["mt"] + params["mh"]
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


def calculate_ct(u_wind, path_ct): 
    """ This function calculates the thrust coefficient (ct)
    for a given wind speed.

    Parameters:
    u_wind(np.ndarray): Wind speed data
    path_ct(str): Path to the Ct file

    Returns:
    float: Ct value
    """
    # load text file that contains data, skip first row since it is header info
    data = np.loadtxt(path_ct, skiprows=1)

    # find mean value of wind
    mean_u = np.mean(u_wind)

    # assign the wind speed and ct values from data to individual variables
    V = data[:, 0]
    CT = data[:, 1]

    # interpolate to find ct for the given wind file
    ct = np.interp(mean_u, V, CT)
    return ct


def calculate_dydt(t, y, M, C, K, rho=None, ct=None,
                   rotor_area=None, t_wind=None, u_wind=None):
    """ This function calculates the derivative of
    the state vector y at time t.

    Parameters:
    t(float): Time
    y(np.ndarray): State vector
    M,C,K(np.ndarray): Mass, damping and stiffness matrices
    rho(float): Air density
    ct(float): Thrust coefficient
    rotor_area(float): Rotor area
    t_wind(np.ndarray): Time for wind speed data
    u_wind(np.ndarray): Wind speed data

    Returns: 
    np.ndarray: Derivative of the state vector y at time t (dy/dt)
    """
    # assemble matrix A =[   O
    #  -inV(M)K -inv(M)C] for O and I we
    # have the degrees of freedom
    # needed for first part: (without force)

    # Define Minv (inverse of matrix M)
    Minv = np.linalg.inv(M)
    # degrees of freedom =2
    ndof = 2
    # Define O and I
    I = np.eye(ndof)  # eye=identity matrix 2x2
    O = np.zeros((ndof, ndof))  # zeros= zeros matrix 2x2
    A = np.block([[O, I], [-Minv @ K, -Minv @ C]])  # A matrix

    if u_wind is None:  # if there is no forcing
        return A@y

    else:  
        # if u_wind is given there is an external force
        # due to wind (Forced response)
        # ensure parameters are provided
        if rho is None or ct is None or rotor_area is None or t_wind is None:
            raise ValueError(f"rho, ct, rotor_area and t_wind must be provided for forced response")
    
        # needed for second part: (with force)
        # assemble B matrix = [   0
        #                     inv(M)*F]
        # 2 degrees of freedom: x1= displacement of blade,
        # x2= displacement of tower, x1*= velocity of blade,
        # x2*=velocity of tower
        v1 = y[2]

        # ensuring that t_wind and u_wind are 1-D arrays
        # t_wind = t_wind[:,:]
        # u_wind = u_wind[:,]

        # interpolation for windspeed at time t (if t is between two values,
        # it interpolates linearly between u_wind values)
        u_t = np.interp(t, t_wind, u_wind)
        # calculate forced response
        f_aero = 0.5*rho*ct*rotor_area*(u_t-v1)*np.abs(u_t-v1)
        # areodynamic force on blades, 0=no external forces on system
        F = np.zeros(ndof)
        F[0] = f_aero
        B = np.zeros(2*ndof)  # initialize the array
        B[ndof:] = Minv @ F
        return A@y+B


def simulate_turbie(path_wind, path_parameters, path_Ct):
    """ This function simulates the response of the turbine system
      to a given wind speed data.

    Parameters:
    path_wind(str): Path to wind speed data
    path_parameters(Str): Path to parameters file
    path_Ct(str): Path to Ct file

    Returns:
    np.ndarray: Time vector (t)
    np.ndarray: Wind speed data (u_wind)
    np.ndarray: Blade deflection (xb)
    np.ndarray: Tower deflection (xt)
    """
    # define our time vector
    t0, tf, dt = 0, 660, 0.01

    # inputs to solve ivp
    tspan = [t0, tf]  # 2-element list of start, stop
    y0 = [0, 0, 0, 0]  # initial condition
    # t_eval = np.arange(t0, tf, dt)  # times at which we want output

    # call functions to get necessary arguments used for solve_ivp
    M, C, K = get_turbie_system_matrices(path_parameters)
    t_wind, u_wind = load_wind(path_wind)
    turbie_params = load_turbie_parameters(path_parameters)

    # specify value of rho and area
    rho = turbie_params['rho']
    area = np.pi * (turbie_params['Dr']/2)**2

    # find ct value
    ct = calculate_ct(u_wind, path_Ct)

    # extra arguments to dydt besides t, y
    args = (M, C, K, ct, rho, area, t_wind, u_wind)
    # run the numerical solver
    res = solve_ivp(calculate_dydt, tspan, y0, t_eval=t_wind, args=args)

    # extract the output
    t, y = res.t, res.y

    # get out relative deflections
    xb = y[0] - y[1]  # relative blade deflection
    xt = y[1]  # tower deflection
    t = t       # time

    return t, u_wind, xb, xt


def save_resp(t, u, xb, xt, path_save):

    """
    Saves time, wind speed, blade deflection, and tower
    deflection data to a text file.

    This function stacks the provided arrays into a two-dimensional
    array and writes them to a file specified by `path_save` in
    tab-delimited format. The file includes a header row labeling
    the columns "Time", "U", "xb", and "xt".

    Parameters
    ----------
    t : array
        Array of time values (e.g., simulation or measurement times).
    u : array
        Array of wind speed measurements corresponding to each time value.
    xb : array
        Array of blade deflection values corresponding to each time value.
    xt : array
        Array of tower deflection values corresponding to each time value.
    path_save : pathlib.Path or str
        Destination path where the output text file will be saved.

    Returns
    -------
    None
        The function writes the data to a file but does not return anything.

    Notes
    -----
    - The output file is tab-delimited (`\\t`) and has a header line:
      "Time \\tU \\txb \\txt".
    - Data are formatted to three decimal places.
    """

    # specify header to contain the associated labels for each column
    header = "Time \tU \txb \txt"

    # stacks 1D arrays as columns into 2D array
    data = np.column_stack((t, u, xb, xt))

    # saves data as a text file to the location specified by path_save
    np.savetxt(path_save, data, delimiter='\t', fmt='%.3f', header=header)


def calculate_mean_stdv(xb, xt, u_wind):

    """
    Calculates the mean and standard deviation of blade and tower deflections,
    as well as the associated mean wind speed.

    Parameters
    ----------
    xb : array
        Array of blade deflection data.
    xt : array
        Array of tower deflection data.
    u_wind : array
        Array of wind speed measurements.

    Returns
    -------
    mean_blade : float
        Mean blade deflection.
    std_blade : float
        Standard deviation of blade deflection.
    mean_tower : float
        Mean tower deflection.
    std_tower : float
        Standard deviation of tower deflection.
    mean_wind : float
        Mean wind speed.
    """

    # use numpy to find mean of the xb array
    mean_blade = np.mean(xb)
    # use numpy to find stdv of the xb array
    std_blade = np.std(xb)

    # use numpy to find mean of the xt array
    mean_tower = np.mean(xt)
    # use numpy to find the stdv of the xt array
    std_tower = np.std(xt)

    # use numpy to find the mean of the wind speed
    mean_wind = np.mean(u_wind)
    return mean_blade, std_blade, mean_tower, std_tower, mean_wind


def calculate_for_TI(path_wind_files, path_ct, turbie_params):

    """
    Calculates and returns the mean and standard deviation of
    blade and tower deflection for multiple wind speed datasets.

    This function:
      1. Reads all `.txt` files in the directory specified by
        `path_wind_files` which each contain data for a specific wind speed.
      2. Sorts them by the wind speed value specified by each file's name
         (the second element when splitting the filename by an underscore).
      3. For each file, calls `simulate_turbie` to obtain time, wind speed,
         blade deflection, and tower deflection data.
      4. Uses `calculate_mean_stdv` to compute the mean and standard
         deviation of both blade and tower deflections.
      5. Collects and returns these statistics for all files.

    Parameters
    ----------
    path_wind_files : pathlib.Path or str
        Directory containing `.txt` files, each representing
        a wind speed dataset.Filenames must include the wind speed
        as the second segment when splitting by underscore (e.g., 'data_12.txt'
        where 12 is the wind speed).
    path_ct : pathlib.Path or str
        Path to a file or directory containing additional parameters needed by
        the `simulate_turbie` function (e.g., turbine thrust coefficient data).
    turbie_params : dict or any
        A dictionary or object holding parameters for the turbine simulation.
        turbie_params must contain the variable 'rho' and the associated
        density of air as well as the variable 'Dr' and the associated
        diameter of the swept area.

    Returns
    -------
    blade_data : list of list
        List of `[mean_wind, mean_blade, std_blade]` for each
        wind speed dataset.
    tower_data : list of list
        List of `[mean_wind, mean_tower, std_tower]` for each
        wind speed dataset.
    """
    # initalize two empty lists (blade_mean_stdv and tower_mean_stdv) to be
    # used to store standard deviations and means for each wind speed data set
    blade_data = []
    tower_data = []

    # Get all .txt files in the folder
    files = [f for f in os.listdir(path_wind_files) if f.endswith('.txt')]

    # Sort the files by extracting the wind speed value
    # (which is the second element when splitting by '_')
    files.sort(key=lambda x: int(x.split('_')[1]))
    for file_name in files:
        path_wind = path_wind_files/(str(file_name))
        #  call simulate_turbie to get time, wind, and deflections
        t, wind, xb, xt = simulate_turbie(path_wind, turbie_params, path_ct)

        # calculate the mean and standard deviation of the blade
        # using calculate_mean_stdv function
        mean_blade, std_blade, mean_tower, std_tower, mean_wind = calculate_mean_stdv(xb, xt, wind)

        # append the mean and stdv values previously calculated to the list
        # in order to store the mean and stdv associated with each wind speed
        blade_data.append([mean_wind, mean_blade, std_blade])
        tower_data.append([mean_wind, mean_tower, std_tower])

    return blade_data, tower_data


def plot_mean_stdv(blade_array, tower_array, TI_title=str):

    """
    Plots the mean deflection and standard deviation for blade
    and tower data in two subplots.

    Each array should be of shape (N, 3), where:
      - The first column represents mean wind speed [m/s].
      - The second column represents mean deflection [m].
      - The third column represents the standard deviation of the deflection.

    Parameters:
    ----------
    blade_array : numpy.ndarray
        2D array containing blade deflection data (wind speed,
        mean deflection, standard deviation).
    tower_array : numpy.ndarray
        2D array containing tower deflection data (wind speed,
        mean deflection, standard deviation).
    TI_title : str, optional
        A string to identify the turbulence intensity (TI)
        or any other descriptor to be included in each subplot title.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the two subplots.
    axs : numpy.ndarray
        An array of Axes objects for the subplots.

    """
    # specifying subplots and figure size
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    # using error bar to plot the stdv in relation
    # to each mean blade deflection value
    axs[0].errorbar(blade_array[:, 0], blade_array[:, 1],
                    yerr=blade_array[:, 2], fmt='o-',
                    capsize=2, label='Blade Deflection')
    # set x label
    axs[0].set_xlabel('Mean Wind Speed [m/s]')
    # set y label
    axs[0].set_ylabel('Blade Deflection [m]')
    # ensuring all x values are displayed on the x axis
    axs[0].set_xticks(blade_array[:, 0])
    # setting title to display the title including the
    # TI value for which data is being plotted
    axs[0].set_title('Blade Deflection vs Mean Wind Speed,' + TI_title)

    axs[1].errorbar(tower_array[:, 0], tower_array[:, 1],
                    yerr=tower_array[:, 2], fmt='o-',
                    color='r', capsize=2, label='Tower Deflection')
    axs[1].set_xlabel('Mean Wind Speed [m/s]')
    axs[1].set_ylabel('Tower Deflection [m]')
    axs[1].set_xticks(tower_array[:, 0])
    axs[1].set_title('Tower Deflection vs Mean Wind Speed,' + TI_title)
    fig.tight_layout()
    return fig, axs
