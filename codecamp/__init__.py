"""Turbie functions."""
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def load_resp(path_resp,t_start=60):
    data = np.loadtxt(path_resp,skiprows=1)
    data = data[data[:,0]>=t_start] 
    #t,u,xb,xt = np.hsplit(data,4)
    t, u, xb, xt = data.T
    return t, u, xb, xt


def load_wind(path_wind,t_start=0):
    data = np.loadtxt(path_wind,skiprows=1)
    data = data[data[:,0]>=t_start] 
    #t_wind,u_wind = np.hsplit(data,2)
    t_wind, u_wind = data.T
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

def load_turbie_parameters(path_parameters):
    data= np.loadtxt(path_parameters, dtype=str, skiprows=1) #load data as string (both numerical values and strings)
    values, keys = np.hsplit(data, [1]) #values get first column, keys get all remaining columns
    values = data[:,0].astype(float) #convert first column to float (values)
    clean_keys= []

    # BELOW: Sorting through each list in the key list, then selecting the value in index position 1 
    # (which associates with the variable name) and assigning only this str to the cleaned_key list
    # so that the cleaned list only contains the associated variable name for each value of the data
    for k in keys: 
        clean_keys.append(k[1])

    turbie_dict = dict(zip(clean_keys, values))
    return turbie_dict



def get_turbie_system_matrices(path_parameters):
    params= load_turbie_parameters(path_parameters)

    # Define mass values
    m1 = 3 * params["mb"]  # Mass of 3 blades
    m2 = params["mn"] + params["mt"] + params["mh"]      # Mass at tower + nacelle + hub
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
    data = np.loadtxt(path_ct,skiprows=1)
    mean_u = np.mean(u_wind)
    V = data[:,0]
    CT = data[:,1]
    ct = np.interp(mean_u,V,CT)
    return ct

def calculate_dydt(t,y,M,C,K,rho=None,ct=None,rotor_area=None,t_wind=None,u_wind=None):
#assemble matrix A =[   O          I
#                    -inV(M)K -inv(M)C] for O and I we have the degrees of freedom  
    #needed for first part: (without force)

    #Define Minv (inverse of matrix M)
    Minv=np.linalg.inv(M)
    #degrees of freedom =2
    ndof=2
    #Define O and I
    I= np.eye(ndof)  #eye=identity matrix 2x2
    O=np.zeros((ndof,ndof)) #zeros= zeros matrix 2x2
    A = np.block([[O, I], [-Minv @ K, -Minv @ C]]) #A matrix
    
    if u_wind is None: #if there is no forcing 
        return A@y
    
    else: #if u_wind is given there is an external force due to wind (Forced response)
         #ensure parameters are provided
        if rho is None or ct is None or rotor_area is None or t_wind is None:
            raise ValueError(f"rho,ct,rotor_area and t_wind must be provided for forced response")
        
        #needed for second part: (with force)
        #assemble B matrix = [   0
        #                     inv(M)*F]
        #2 degrees of freedom: x1= displacement of blade, x2= displacement of tower, x1*= velocity of blade, x2*=velocity of tower
        v1=y[2]

        # ensuring that t_wind and u_wind are 1-D arrays 
        # t_wind = t_wind[:,:]
        # u_wind = u_wind[:,]

        #interpolation for windspeed at time t (if t is between two values, it interpolates linearly between u_wind values)
        u_t=np.interp(t,t_wind, u_wind)
        #calculate forced response
        f_aero= 0.5*rho*ct*rotor_area*(u_t-v1)*np.abs(u_t-v1)
        F=np.zeros(ndof) #areodynamic force on blades, 0=no external forces on system
        F[0]=f_aero
        B = np.zeros(2*ndof)  # initialize the array
        B[ndof:] = Minv @ F
        shapeAY = np.shape(A@y)
        shapeB = np.shape(B)
        
        return A@y+B 


def simulate_turbie(path_wind,path_parameters,path_Ct):
    # define our time vector
    t0, tf, dt = 0, 660, 0.01

    # inputs to solve ivp
    tspan = [t0, tf]  # 2-element list of start, stop
    y0 = [0, 0, 0, 0]  # initial condition
    t_eval = np.arange(t0, tf, dt)  # times at which we want output

    M,C,K = get_turbie_system_matrices(path_parameters)
    t_wind, u_wind = load_wind(path_wind)
    turbie_params = load_turbie_parameters(path_parameters)
    rho = turbie_params['rho']
    area = np.pi * (turbie_params['Dr']/2)**2
    ct = calculate_ct(u_wind,path_Ct)

    args = (M, C, K, ct, rho, area, t_wind, u_wind)  # extra arguments to dydt besides t, y
    # run the numerical solver
    res = solve_ivp(calculate_dydt, tspan, y0, t_eval=t_wind, args=args)

    # extract the output
    t, y = res.t, res.y

    # get out relative deflections
    xb = y[0] - y[1]  # relative blade deflection
    xt = y[1]  # tower deflection
    t = t
    
    return t, u_wind, xb, xt

def save_resp(t,u,xb,xt,path_save):
    header="Time \tU \txb \txt"
    data =np.column_stack((t,u,xb,xt)) #stacks 1D arrays as columns into 2D array
    np.savetxt(path_save, data, delimiter='\t', fmt='%.3f', header=header)

