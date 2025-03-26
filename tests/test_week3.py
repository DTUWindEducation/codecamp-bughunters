"""Check that your Week 3 functions work as expected.

Note that these tests are NOT exhaustive. But they should provide some basic
checks that what you are doing is correct.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import codecamp


DATA_DIR = Path('./data')


def test_load_resp_default():
    """Check default load_resp value."""
    # given
    path_resp_file = DATA_DIR / 'resp_12_ms_TI_0.1.txt'
    xt_exp_tstart = 0.400  # expected value of xb at 70 seconds
    tstart_exp = 60  # expected default tstart
    # when
    t, _, _, xt = codecamp.load_resp(path_resp_file)
    # then
    assert np.isclose(t[0], tstart_exp)  # check tstart is right
    assert np.isclose(xt[0], xt_exp_tstart)  # also check value of xt


def test_load_resp():
    """Check that load_resp correctly loads file with tstart, including type."""
    # given
    path_resp_file = DATA_DIR / 'resp_12_ms_TI_0.1.txt'
    t_start = 70  # start value
    xb_exp_70s = 0.594  # expected value of xb at 70 seconds
    # when
    t, _, xb, _ = codecamp.load_resp(path_resp_file, t_start=t_start)
    # then
    assert np.isclose(t[0], t_start)  # check tstart is right
    assert np.isclose(xb[0], xb_exp_70s)  # also check value of xb
    assert isinstance(t, np.ndarray)  # make sure t is a numpy array


def test_load_wind_default():
    """Check the load_wind function with default values."""
    # given
    path_wind_file = DATA_DIR / 'wind_12_ms_TI_0.1.txt'
    u_exp_tstart = 12.578  # expected value of v at 0 seconds
    tstart_exp = 0  # expected default tstart
    # when
    t_wind, u_wind = codecamp.load_wind(path_wind_file)
    # then
    assert np.isclose(t_wind[0], tstart_exp)  # check tstart is right
    assert np.isclose(u_wind[0], u_exp_tstart)  # also check value of xt


def test_load_wind_tstart():
    """Check the load_wind function with tstart."""
    # given
    path_wind_file = DATA_DIR / 'wind_12_ms_TI_0.1.txt'
    tstart = 50  # expected default tstart
    u_exp_50s = 11.969  # expected value of v at 50 seconds
    # when
    t_wind, u_wind = codecamp.load_wind(path_wind_file, t_start=tstart)
    # then
    assert np.isclose(t_wind[0], tstart)  # check tstart is right
    assert np.isclose(u_wind[0], u_exp_50s)  # also check value of xt
    assert isinstance(t_wind, np.ndarray)  # make sure t is a numpy array


def test_plot_resp(monkeypatch):  # use a pytest "monkeypatch" to stop plots from popping up
    """Check some aspects of plot_resp"""
    monkeypatch.setattr(plt, 'show', lambda: None)  # temporarily overwrite plt.show() to do nothing
    # given
    n_sbplts_exp = 2  # number of expected subplots
    xlim_exp = (60, 660)
    path_resp_file = DATA_DIR / 'resp_12_ms_TI_0.1.txt'
    t, v, xb, xt = codecamp.load_resp(path_resp_file)
    # when
    fig, axs = codecamp.plot_resp(t, v, xb, xt)
    # then
    assert isinstance(fig, plt.Figure)  # check it's a figure
    assert isinstance(axs, tuple)  # check it's a tuple
    assert len(axs) == n_sbplts_exp  # check there are 2 subplots
    assert axs[0].get_xlim() == xlim_exp  # check the xlim is right


def test_load_turbie_parameters():
    """Check load_turbie_params works."""
    # given
    path_param_file = DATA_DIR / 'turbie_parameters.txt'
    n_exp_keys = 14  # number of expected keys in dictionary
    val_fb = 0.63  # value of the fb variable
    # when
    turbie_params = codecamp.load_turbie_parameters(path_param_file)
    # then
    assert len(turbie_params) == n_exp_keys
    assert np.isclose(val_fb, turbie_params['fb'])


def test_get_turbie_system_matrices():
    """Check values of get_turbie_system_matrices()"""
    # given
    path_param_file = DATA_DIR / 'turbie_parameters.txt'
    M_exp = np.array([[123000, 0], [0, 1179000]])
    C_exp = np.array([[4208, -4208], [-4208, 16938]])
    K_exp = np.array([[1711000, -1711000], [-1711000, 4989000]])
    # when
    M, C, K = codecamp.get_turbie_system_matrices(path_param_file)
    # then
    np.testing.assert_array_equal(M, M_exp)
    np.testing.assert_array_equal(C, C_exp)
    np.testing.assert_array_equal(K, K_exp)
