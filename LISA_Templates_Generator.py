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

# 设定agg.path.chunksize值，例如设置为10000
mpl.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from functools import partial
from scipy import signal
from scipy.fftpack import fft, ifft
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import yaml

# set up sampling parameters
# read parameters from config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

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
Tc = config["Tc"] * T  # chirp time

Phic = config["Phic"]  # chirp phase
ThetaS = config["ThetaS"]  # sky location
PhiS = config["PhiS"]  # sky location
Iota = config["Iota"]  # inclination
Psi = config["Psi"]  # polarization
# ThetaS = np.arccos(np.random.uniform(-1,1))
# PhiS   = np.random.uniform(0,2*np.pi)
# Iota   = np.arccos(np.random.uniform(-1,1))
# Psi    = np.random.uniform(0,2*np.pi)
# print(ThetaS,PhiS,Iota,Psi)

Z = 10  # cosmological redshift
M1sun = config["M1sun"]  # solar mass as unit
M2sun = config["M2sun"]  # solar mass as unit
# Tobs   = 0.5   # year as unit
# Chi1   = 0.1    # dimensionless parameter
# Chi2   = 0.0    # dimensionless parameter
cosmo = FlatLambdaCDM(H0=67, Om0=0.32)
DL = cosmo.luminosity_distance(Z).value * MpcInS  # Mpc in second
M1 = (1 + Z) * M1sun * MsunInS  # solar mass in second
M2 = (1 + Z) * M2sun * MsunInS  # solar mass in second
M = M1 + M2  # total mass
Qmr = M1 / M2  # mass ratio
Mu = M1 * M2 / M  # reduced mass
Mc = Mu ** (3.0 / 5) * M ** (2.0 / 5)  # chirp mass
Eta = M1 * M2 / M**2  # symmetric mass ratio
tc_true = Tc
phic_true = Phic
mc_true = Mc
eta_true = Eta
dl_true = DL
thetas_true = ThetaS
phis_true = PhiS
iota_true = Iota
psi_true = Psi


def ht_respon_TQ(t, tc, phic, mc, eta, dl, thetaS, phiS, iota, psi):
    """
    For TianQin (one Michelson interferometer): (thetaS,phiS) is location of source,
    (thJ,phJ) is latitude and longitude of J0806 in heliocentric-ecliptic frame
    """

    thJ = 1.65273
    phJ = 2.10213
    kap = 2 * np.pi / OrbitPeriodInS * t
    Dplus_TQ = (
        np.sqrt(3.0)
        / 32
        * (
            8
            * np.cos(2 * kap)
            * (
                (3 + np.cos(2 * thetaS)) * np.sin(2 * (phiS - phJ)) * np.cos(thJ)
                + 2 * np.sin(thJ) * np.sin(phiS - phJ) * np.sin(2 * thetaS)
            )
            - 2
            * np.sin(2 * kap)
            * (
                3
                + np.cos(2 * (phiS - phJ))
                * (9 + np.cos(2 * thetaS) * (3 + np.cos(2 * thJ)))
                - 6 * np.cos(2 * thJ) * (np.sin(phiS - phJ)) ** 2
                - 6 * np.cos(2 * thetaS) * (np.sin(thJ)) ** 2
                + 4 * np.cos(phiS - phJ) * np.sin(2 * thJ) * np.sin(2 * thetaS)
            )
        )
    )

    Dcros_TQ = (
        np.sqrt(3.0)
        / 4
        * (
            -4
            * np.cos(2 * kap)
            * (
                np.cos(2 * (phiS - phJ)) * np.cos(thJ) * np.cos(thetaS)
                + np.cos(phiS - phJ) * np.sin(thetaS) * np.sin(thJ)
            )
            + np.sin(2 * kap)
            * (
                np.cos(thetaS) * (3 + np.cos(2 * thJ)) * np.sin(2 * (phJ - phiS))
                + 2 * np.sin(phJ - phiS) * np.sin(thetaS) * np.sin(2 * thJ)
            )
        )
    )

    # """antenna pattern function for '+' mode"""
    Fplus_TQ = (np.cos(2 * psi) * Dplus_TQ - np.sin(2 * psi) * Dcros_TQ) / 2.0

    # """antenna pattern function for '×' mode"""
    Fcros_TQ = (np.sin(2 * psi) * Dplus_TQ + np.cos(2 * psi) * Dcros_TQ) / 2.0

    # """1st MI angular response function for TianQin"""
    Q_TQ = np.sqrt(
        (1 + (np.cos(iota)) ** 2) ** 2 / 4 * (Fplus_TQ) ** 2
        + (np.cos(iota)) ** 2 * (Fcros_TQ) ** 2
    )

    # """1st MI polarization phase for TianQin"""
    phip_TQ = -np.arctan(
        2 * np.cos(iota) * Fcros_TQ / ((1 + (np.cos(iota)) ** 2) * Fplus_TQ)
    )

    # """non-precesnp.sing spinning PN correction to the orbital phase"""
    THETA = eta * (tc - t) / (5 * mc / eta ** (3.0 / 5))
    PSI_PN = phic - THETA ** (5.0 / 8) / eta * (
        1
        + (3715.0 / 8064 + 55.0 / 96 * eta) * THETA ** (-1.0 / 4)
        - 3 * np.pi / 4 * THETA ** (-3.0 / 8)
        + (9275495.0 / 14450688 + 284875.0 / 258048 * eta + 1855.0 / 2048 * eta**2)
        * THETA ** (-1.0 / 2)
    )

    # """the frequency-domain amplitude: 1st 60 degree MI with antenna response (contain plus and cross mode) for TianQin"""
    ht_TQ = (
        -mc
        * Q_TQ
        / dl
        * ((tc - t) / (5 * mc)) ** (-1.0 / 4)
        * np.cos(phip_TQ + 2 * PSI_PN)
    )
    #     ht_TQ = np.concatenate([ht_TQ, len(t)*[0]])
    return ht_TQ


def ht_respon_LISA(t, tc, phic, mc, eta, dl, thetaS, phiS, iota, psi):
    # set up constants
    c = 1
    G = 1

    """For LISA (one Michelson interferometer): (thetaS,phiS) is location of source"""
    #     alpha = 2*np.pi/YearInS* t-np.pi/9 # trailing 20°behind the Earth (-np.pi/9)
    alpha = 2 * np.pi / YearInS * t
    lam = 3 * np.pi / 4
    Dplus_LISA = (
        np.sqrt(3.0)
        / 64
        * (
            -36 * (np.sin(thetaS)) ** 2 * np.sin(2 * alpha - 2 * lam)
            + (3 + np.cos(2 * thetaS))
            * (
                np.cos(2 * phiS) * (9 * np.sin(2 * lam) - np.sin(4 * alpha - 2 * lam))
                + np.sin(2 * phiS) * (np.cos(4 * alpha - 2 * lam) - 9 * np.cos(2 * lam))
            )
            - 4
            * np.sqrt(3.0)
            * np.sin(2 * thetaS)
            * (np.sin(3 * alpha - 2 * lam - phiS) - 3 * np.sin(alpha - 2 * lam + phiS))
        )
    )

    Dcros_LISA = (
        1
        / 16
        * (
            np.sqrt(3.0)
            * np.cos(thetaS)
            * (9 * np.cos(2 * lam - 2 * phiS) - np.cos(4 * alpha - 2 * lam - 2 * phiS))
            - 6
            * np.sin(thetaS)
            * (np.cos(3 * alpha - 2 * lam - phiS) + 3 * np.cos(alpha - 2 * lam + phiS))
        )
    )

    # """antenna pattern function for '+' mode"""
    Fplus_LISA = (np.cos(2 * psi) * Dplus_LISA - np.sin(2 * psi) * Dcros_LISA) / 2.0

    # """antenna pattern function for '×' mode"""
    Fcros_LISA = (np.sin(2 * psi) * Dplus_LISA + np.cos(2 * psi) * Dcros_LISA) / 2.0

    # """1st MI angular response function for TianQin"""
    Q_LISA = np.sqrt(
        (1 + (np.cos(iota)) ** 2) ** 2 / 4 * (Fplus_LISA) ** 2
        + (np.cos(iota)) ** 2 * (Fcros_LISA) ** 2
    )

    # """1st MI polarization phase for TianQin"""
    phip_LISA = -np.arctan(
        2 * np.cos(iota) * Fcros_LISA / ((1 + (np.cos(iota)) ** 2) * Fplus_LISA)
    )

    # """non-precesnp.np.sing spinning PN correction to the GW phase"""
    THETA = eta * (tc - t) / (5 * mc / eta ** (3.0 / 5))
    PSI_PN = phic - THETA ** (5.0 / 8) / eta * (
        1
        + (3715.0 / 8064 + 55.0 / 96 * eta) * THETA ** (-1.0 / 4)
        - 3 * np.pi / 4 * THETA ** (-3.0 / 8)
        + (9275495.0 / 14450688 + 284875.0 / 258048 * eta + 1855.0 / 2048 * eta**2)
        * THETA ** (-1.0 / 2)
    )

    """the frequency-domain amplitude: 1st 60 degree MI with antenna response (contain plus and cross mode) for TianQin"""
    ht_LISA = (
        -mc
        * Q_LISA
        / dl
        * ((tc - t) / (5 * mc)) ** (-1.0 / 4)
        * np.cos(phip_LISA + 2 * PSI_PN)
    )
    # m is the total mass, express m in terms of mc and eta
    m = mc / eta ** (3 / 5)
    f_iso = c**3 / (G * m * np.pi * 6 ** (3 / 2))  # 最内稳定圆轨道的引力波频率
    tau = c**3 * eta / (5 * G * m) * (tc - t)
    w = (
        c**3
        / (8 * G * m)
        * (
            tau ** (-3 / 8)
            + (743 / 2688 + 11 / 32 * eta) * tau ** (-5 / 8)
            - 3 * np.pi / 10 * tau ** (-3 / 4)
            + (1855099 / 14450688 + 56975 / 258048 * eta + 371 / 2048 * eta**2)
            * tau ** (-7 / 8)
        )
    )
    # print(f'max orbital frequency: {max(w)}, innermost stable circular orbit frequency: {f_iso*np.pi}')
    # if any(x >= f_iso*np.pi for x in w):
    #     print(f'Orbital frequency exceeds the innermost stable circular orbit frequency, cutoff at {f_iso*np.pi} Hz.')
    #     index = next(i for i, x in enumerate(w) if x >= f_iso*np.pi)
    #     ht_LISA[index:-1]=0
    # find the index of the max(w)
    index = np.argmax(w)
    ht_LISA[index - 5 :] = 0

    return ht_LISA


# def ht_model(t, tc, phic, mc, eta, dl, thetaS, phiS, iota, psi):
#     ht = np.piecewise(t, [t >= tc, t < tc],
#                       [0, partial(ht_respon_TQ, tc=tc, phic=phic, mc=mc, eta=eta, dl=dl, thetaS=thetaS, phiS=phiS, iota=iota, psi=psi)])
#     return ht


def ht_model(t, tc, phic, mc, eta, dl, thetaS, phiS, iota, psi):
    ht = np.piecewise(
        t,
        [t >= tc, t < tc],
        [
            0,
            partial(
                ht_respon_LISA,
                tc=tc,
                phic=phic,
                mc=mc,
                eta=eta,
                dl=dl,
                thetaS=thetaS,
                phiS=phiS,
                iota=iota,
                psi=psi,
            ),
        ],
    )
    return ht


def TQPSD(f):
    """CQG 2018"""
    Sx = 1e-24
    Sa = 1e-30
    L0 = np.sqrt(3.0) * 1e5 * 1e3
    return Sx / (L0**2) + 4 * Sa / ((2 * np.pi * f) ** 4 * L0**2) * (1.0 + 1e-4 / f)


def LISAPSD(f):
    """MCMC of SMBHB, Cornish and Porter, CQG 2006"""
    Spos = 4e-22
    Sacc = 9e-30
    L = 5e9
    return (4 * Spos + 16 * Sacc / (2 * np.pi * f) ** 4) / (4 * L**2)


# calculate the SNR
def SNR(data, T, fs, detector="LISA"):
    """
    计算数据Data的单边功率谱密度:
    T: 数据的时间长度
    fs: 数据的采样频率
    detector: 天琴或者 LISA
    """
    if detector == "TQ":
        PSD = TQPSD
    elif detector == "LISA":
        PSD = LISAPSD
    else:
        print("PSD must be TQ or LISA")
        return 0
    N = len(data)
    delta_f = 1 / T
    # f = np.arange(0, fs / 2, delta_f)
    f = np.linspace(0, fs / 2, N // 2 + 1)  # frequency vector
    # print(f'total number of points {N}, length of frequency {len(f)}')
    psd = PSD(f[1:])
    xf = np.fft.fft(data)
    absf = np.abs(xf)[1 : N // 2 + 1] / fs  # single-sided spectrum
    # print(f'length of PSD {len(psd)}\n length of Pxx {len(Pxx)}')
    SNR = np.sqrt(4 * np.sum(absf**2 / psd * delta_f))
    return SNR


# main function
def main(t=t):
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
    snr = config["SNR"]
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
        "fig/whitened-data-signal-fs"
        + str(fs)
        + "-SNR"
        + "{:.2f}".format(snr_lisa)
        + ".png",
        dpi=300,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Process Done in {time.time()-start_time:.2f}s")
