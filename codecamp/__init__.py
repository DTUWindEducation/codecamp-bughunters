"""Turbie functions.
"""
from pathlib import Path
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def load_resp(path_resp,t_start=60):
    data = np.loadtxt(path_resp,skiprows=1)
    data = data[data[:,0]>=t_start] 
    t,u,xb,xt = np.hsplit(data,4)
    return t, u, xb, xt


def load_wind(path_wind,t_start=0):
    data = np.loadtxt(path_wind,skiprows=1)
    data = data[data[:,0]>=t_start] 
    t_wind,u_wind = np.hsplit(data,2)
    return t_wind, u_wind


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
        t_wind = t_wind[:,0]
        u_wind = u_wind[:,0]

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
    args = (M, C, K, ct, rho, area, t_wind, wind)  # extra arguments to dydt besides t, y

    # run the numerical solver
    res = solve_ivp(calculate_dydt, tspan, y0, t_eval=t_eval, args=args)

    # extract the output
    t, y = res.t, res.y

    return t, y

DATA_DIR = Path('./data')
path_wind_file = DATA_DIR / 'wind_12_ms_TI_0.1.txt'
path_param_file = DATA_DIR / 'turbie_parameters.txt'
path_ct_file = DATA_DIR / 'CT.txt'
path_resp_file = DATA_DIR / 'resp_12_ms_TI_0.1.txt'
t_exp, u_exp, xb_exp, xt_exp = load_resp(path_resp_file, t_start=0)
_, u_wind = load_wind(path_wind_file, t_start=0)

print(simulate_turbie(path_wind_file, path_param_file, path_ct_file))
#print(np.shape(t),np.shape(u2))
