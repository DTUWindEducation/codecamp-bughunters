"""Test for your Week 4 functions."""
from pathlib import Path

import numpy as np

import codecamp


DATA_DIR = Path('./data')


def test_calculate_Ct():
    """Check interpolation works."""
    # given
    path_ct_file = DATA_DIR / 'CT.txt'
    u = 4.5
    ct_exp = 0.921
    # when
    ct = codecamp.calculate_ct(u, path_ct_file)
    # then
    assert np.isclose(ct, ct_exp)


def test_dydt_homogeneous():
    """Check homogenous solution, including shape"""
    # given
    path_param_file = DATA_DIR / 'turbie_parameters.txt'
    t, y = 1, [1, 2, 3, 4]
    M, C, K = codecamp.get_turbie_system_matrices(path_param_file)
    dydt_exp = np.array([3., 4., 13.94478049, -7.05863274])
    shape_exp = (M.shape[0]*2,)  # expected shape of dydt
    # when
    dydt = codecamp.calculate_dydt(t, y, M, C, K)
    # then
    assert isinstance(dydt, np.ndarray)  # check it's a numpy array
    assert dydt.shape == shape_exp  # check shape is correct
    np.testing.assert_allclose(dydt, dydt_exp)  # check the numbers are correct


def test_dydt_forced():
    """Check forced solution, including shape"""
    # given
    path_param_file = DATA_DIR / 'turbie_parameters.txt'
    path_wind_file = DATA_DIR / 'wind_12_ms_TI_0.1.txt'
    t, y = 1, [1, 2, 3, 4]
    turbie_params = codecamp.load_turbie_parameters(path_param_file)
    M, C, K = codecamp.get_turbie_system_matrices(path_param_file)
    t_wind, u_wind = codecamp.load_wind(path_wind_file, t_start=0)
    rho = turbie_params['rho']
    rotor_area = np.pi * (turbie_params['Dr']/2)**2
    ct = 0.5770002944545457
    dydt_exp = np.array([ 3, 4, 20.44459732, -7.05863274])
    shape_exp = (M.shape[0]*2,)  # expected shape of dydt
    # when
    dydt = codecamp.calculate_dydt(t, y, M, C, K, rho=rho, ct=ct,
                                   rotor_area=rotor_area, t_wind=t_wind, u_wind=u_wind)
    # then
    assert isinstance(dydt, np.ndarray)  # check it's a numpy array
    assert dydt.shape == shape_exp  # check shape is correct
    np.testing.assert_allclose(dydt, dydt_exp)  # check the numbers are correct


def test_simulate_turbie():
    """Check that our simulated response to wind matches test file"""
    # given
    path_wind_file = DATA_DIR / 'wind_12_ms_TI_0.1.txt'
    path_param_file = DATA_DIR / 'turbie_parameters.txt'
    path_ct_file = DATA_DIR / 'CT.txt'
    path_resp_file = DATA_DIR / 'resp_12_ms_TI_0.1.txt'
    t_exp, u_exp, xb_exp, xt_exp = codecamp.load_resp(path_resp_file, t_start=0)
    _, u_wind = codecamp.load_wind(path_wind_file, t_start=0)
    # when
    t, u2, xb, xt = codecamp.simulate_turbie(path_wind_file, path_param_file, path_ct_file)
    # then
    np.testing.assert_allclose(u_wind, u_exp)  # check the wind in the resp. matches wind in file
    np.testing.assert_allclose(u2, u_exp)  # check wind returned by sim turbie matches wind in file
    np.testing.assert_allclose(t, t_exp)  # check the time is the same
    np.testing.assert_allclose(xb, xb_exp, atol=1.1e-2)
    np.testing.assert_allclose(xt, xt_exp, atol=1.1e-2)


def test_save_resp(tmp_path):  # use pytest to create a temporary directory for our test file
    """If we load, save and reload, should be identical values."""
    # given
    path_load_file = DATA_DIR / 'resp_12_ms_TI_0.1.txt'  # files to load from
    path_save_file = tmp_path / 'test.txt'  # file to save to
    t, u, xb, xt = codecamp.load_resp(path_load_file, t_start=0)  # load values
    # when
    codecamp.save_resp(t, u, xb, xt, path_save_file)  # call our save function
    t_rl, u_rl, xb_rl, xt_rl = codecamp.load_resp(path_save_file, t_start=0)  # reload the values
    # then
    np.testing.assert_array_equal(t, t_rl)  # check the original and reloaded values
    np.testing.assert_array_equal(u, u_rl)  # match exactly
    np.testing.assert_array_equal(xb, xb_rl)
    np.testing.assert_array_equal(xt, xt_rl)
