#!/Users/qyq/miniconda3/envs/data/bin/python
"""
This script generates waveform templates for LISA.
Modified from WenFan Feng's code.
Aknowledge Yao Fu's helpful discussion.
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from LISA_TianQin_Waveforms import *

# 设定agg.path.chunksize值，例如设置为10000
mpl.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from functools import partial
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.stats import loguniform
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import yaml
import h5py

# set up sampling parameters
# read parameters from config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    YearInS = (1 * u.yr).to(u.s).value  # one year in [s]
    fs = config["fs"]  # sampling frequency
    T = int(config["T"] * YearInS)  # Total time
    N = int(T * fs)  # Number of data length
    t = np.linspace(0, T, N)  # time vector
    # calculate f
    df = 1 / T  # frequency resolution
    # f = np.arange(0,fs/2,df) # frequency vector
    f = np.linspace(0, fs / 2, N // 2 + 1)  # frequency vector
    print(f"number of f {len(f)}, t {len(t)/2}")

    """
    mutable paramters: Phic, ThetaS, Phis, Iota, Psi, M1, M2, Tc
    """
    # convert units from SI to Natural units
    MsunInS = (
        (const.M_sun * const.G / const.c**3).to(u.s).value
    )  # 4.926860879228572e-06   # solar mass in [s] GM/C^3
    MpcInS = (
        (1 * u.Mpc / const.c).to(u.s).value
    )  # 102927125053532.6       # mega parsec in [s]
    OrbitRadiusInS = 1e8 / const.c.value  # 1e5 km
    MearthInS = const.M_earth.value * const.G.value / const.c.value**3
    OrbitPeriodInS = 2 * np.pi * np.sqrt(OrbitRadiusInS**3 / MearthInS)
    AUInS = const.au.value / const.c.value  # AU in [s]

    Tc_min, Tc_max = config["Tc"]  # chirp time
    Phic_min, Phic_max = config["Phic"]  # chirp phase
    ThetaS_min, ThetaS_max = config["ThetaS"]  # sky location
    PhiS_min, PhiS_max = config["PhiS"]  # sky location
    Iota_min, Iota_max = config["Iota"]  # inclination
    Psi_min, Psi_max = config["Psi"]  # polarization

    Z = 10  # cosmological redshift
    M1sun_min, M1sun_max = config["M1sun"]  # solar mass as unit
    M2sun_min, M2sun_max = config["M2sun"]  # solar mass as unit
    # Tobs   = 0.5   # year as unit
    # Chi1   = 0.1    # dimensionless parameter
    # Chi2   = 0.0    # dimensionless parameter
    cosmo = FlatLambdaCDM(H0=67, Om0=0.32)
    DL = cosmo.luminosity_distance(Z).value * MpcInS  # Mpc in second
    snr = config["SNR"]
    cnt = 0 # cnt for waveform number
    for _ in range(1): # 10 Tc
        Tc = np.random.uniform(Tc_min, Tc_max) * T
        for _ in range(1): # 50 (M1, M2)
            M1sun = loguniform.rvs(M1sun_min, M1sun_max)
            M2sun = loguniform.rvs(M2sun_min, M2sun_max)
            M1 = (1 + Z) * M1sun * MsunInS  # solar mass in second
            M2 = (1 + Z) * M2sun * MsunInS  # solar mass in second
            M = M1 + M2  # total mass
            Qmr = M1 / M2  # mass ratio
            Mu = M1 * M2 / M  # reduced mass
            Mc = Mu ** (3.0 / 5) * M ** (2.0 / 5)  # chirp mass
            Eta = M1 * M2 / M**2  # symmetric mass ratio
            for _ in range(1): # 100 (ThetaS, Phis)
                ThetaS = np.arccos(np.random.uniform(ThetaS_min, ThetaS_max))
                PhiS   = np.random.uniform(PhiS_min, PhiS_max)
                for _ in range(1): # 10 Iota
                    Iota   = np.arccos(np.random.uniform(Iota_min, Iota_max))
                    for _ in range(1): # 10 Psi
                        Psi    = np.random.uniform(Psi_min, Psi_max)
                        for _ in range(1): # 10 Phic
                            Phic = np.random.uniform(Phic_min, Phic_max)
                            for snr_ in snr:
                                generate_waveform(T, t, f, fs, N, Tc, Phic, Mc, Eta, DL, ThetaS, PhiS, Iota, Psi, snr_, cnt)
                                cnt += 1
                                with open("sim.log", "a") as log_file:
                                    log_file.write(f"Tc: {Tc}, Phic: {Phic}, Mc: {Mc}, Eta: {Eta}, DL: {DL}, ThetaS: {ThetaS}, PhiS: {PhiS}, Iota: {Iota}, Psi: {Psi}, SNR: {snr_}\n")


# generate waveform
def generate_waveform(T, t, f, fs, N, tc_true, phic_true, mc_true, eta_true, dl_true, thetas_true, phis_true, iota_true, psi_true, snr, cnt=0):
    '''
    Generate waveform with given parameters
    Args:
        T: total time
        t: time vector
        f: frequency vector
        fs: sampling frequency
        N: number of data length
        tc_true: chirp time
        phic_true: chirp phase
        mc_true: chirp mass
        eta_true: symmetric mass ratio
        dl_true: luminosity distance
        thetas_true: sky location
        phis_true: sky location
        iota_true: inclination
        psi_true: polarization
        snr: signal-to-noise ratio
        cnt: waveform number
    '''
    # calculate PSD and convert it to two-sided PSD
    PSD = LISAPSD(f)
    PSD[0] = 0  # set the DC part to zero
    PSD2 = (
        np.concatenate([PSD, np.conj(PSD[N % 2 - 2 : 0 : -1])]) / 2
    )  # double-side PSD

    print(
        f"length of PSD {len(PSD)}, length of two-sided {len(PSD2)}, total number of points {N}"
    )

    # generate a white noise realization
    w_noise = np.random.randn(len(PSD2)) * np.sqrt(fs)
    # calculate the noise spectrum
    fft_w_noise = np.fft.fft(w_noise)
    # print(f'length of fft_noise {len(fft_w_noise)}')
    # multiply the spectrum by the square root of the PSD
    fft_noise = fft_w_noise * np.sqrt(PSD2)
    # calculate the inverse FFT
    noise = np.real(np.fft.ifft(fft_noise))

    h_t = ht_model(
        t,
        tc_true,
        phic_true,
        mc_true,
        eta_true,
        dl_true,
        thetas_true,
        phis_true,
        iota_true,
        psi_true,
    )
    Data = h_t + noise

    snr_lisa = SNR(data=h_t, T=T, fs=fs, detector="LISA")
    print(f"SNR of signal: {snr_lisa:.2f}")
    # rescale SNR
    print(f"Desired SNR is {snr}")
    scale = snr / snr_lisa
    dl_true_rs = dl_true / scale
    # re-calculate the h_t
    h_t = ht_model(
        t,
        tc_true,
        phic_true,
        mc_true,
        eta_true,
        dl_true_rs,
        thetas_true,
        phis_true,
        iota_true,
        psi_true,
    )
    Data = h_t + noise
    snr_lisa = SNR(data=h_t, T=T, fs=fs, detector="LISA")
    print(f"Re-scaled SNR of signal: {snr_lisa:.2f}")

    # whiten Data
    fft_Data = np.fft.fft(Data)
    # rfft_Data = np.fft.rfft(Data)
    # print(f'Length of FFT {len(fft_Data)}, length of rFFT {len(rfft_Data)}')
    white_Data = fft_Data / np.sqrt(PSD2)
    # set the first and last element to zero
    white_Data[0] = 0
    # white_Data[-1] = 0
    White_Data_t = np.real(np.fft.ifft(white_Data))
    # drop the first and last 1% of the data
    White_Data_t = White_Data_t[int(0.01 * N) : int(0.99 * N)]
    t = t[int(0.01 * N) : int(0.99 * N)]

    # Save white_h_t and parameters to HDF5
    with h5py.File("waveforms.h5", "a") as fn:
        grp = fn.create_group(f"waveform_{cnt}")
        grp.create_dataset("white_h_t", data=White_Data_t)
        grp.attrs["tc_true"] = tc_true
        grp.attrs["phic_true"] = phic_true
        grp.attrs["mc_true"] = mc_true
        grp.attrs["eta_true"] = eta_true
        grp.attrs["dl_true"] = dl_true
        grp.attrs["thetas_true"] = thetas_true
        grp.attrs["phis_true"] = phis_true
        grp.attrs["iota_true"] = iota_true
        grp.attrs["psi_true"] = psi_true
        grp.attrs["snr"] = snr
    # whiten the signal h_t
    fft_h_t = np.fft.fft(h_t)
    white_h_t = fft_h_t / np.sqrt(PSD2)
    # set the first and last element to zero
    white_h_t[0] = 0
    # white_h_t[-1] = 0
    White_h_t = np.real(np.fft.ifft(white_h_t))
    # drop the first and last 1% of the data
    White_h_t = White_h_t[int(0.01 * N) : int(0.99 * N)]

    # # whiten colored noise
    # fft_Noise = np.fft.fft(noise)
    # white_Noise = fft_Noise/np.sqrt(PSD2)
    # # set the first and last element to zero
    # white_Noise[0] = 0
    # # white_Noise[-1] = 0
    # White_Noise_t = np.real(np.fft.ifft(white_Noise))

    # Plot
    # check if folder fig exists
    if not os.path.exists("fig"):
        os.makedirs("fig")
    # plot the data and the signal
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)
    ax.plot(t, White_Data_t, label="Data")
    ax.plot(t, White_h_t, label="signal")
    ax.set_title("Whitened Data & Signal, SNR = %.2f" % snr_lisa)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        f"fig/{cnt}.png",
        dpi=300,
    )



if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Process Done in {time.time()-start_time:.2f}s")
