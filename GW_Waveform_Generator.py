#!/Users/qyq/miniconda3/envs/psr/bin/python
"""
　　┏┓　　　┏┓+ +
　┏┛┻━━━┛┻┓ + +
　┃　　　　　　　┃ 　
　┃　　　━　　　┃ ++ + + +
 ████━████ ┃+
　┃　　　　　　　┃ +
　┃　　　┻　　　┃
　┃　　　　　　　┃ + +
　┗━┓　　　┏━┛
　　　┃　　　┃　　　　　　　　　　　
　　　┃　　　┃ + + + +
　　　┃　　　┃
　　　┃　　　┃ +  神兽保佑
　　　┃　　　┃    代码无bug　　
　　　┃　　　┃　　+　　　　　　　　　
　　　┃　 　　┗━━━┓ + +
　　　┃ 　　　　　　　┣┓
　　　┃ 　　　　　　　┏┛
　　　┗┓┓┏━┳┓┏┛ + + + +
　　　　┃┫┫　┃┫┫
　　　　┗┻┛　┗┻┛+ + + +


Author: Yiqian Qian
Description: file content
Date: 2020-12-25 16:45:55
LastEditors: Yiqian Qian
LastEditTime: 2021-01-21 00:01:55
FilePath: /undefined/Users/qyq/Research/DNN_for_GW_Parameter_Space_Estimation/GW_Waveform_Generator.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy import signal
from scipy.fftpack import fft, ifft
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import time


def main():
    """
    GW waveform template generator
    Constants set up
    mutable paramters: Phic, ThetaS, Phis, Iota, Psi, M1, M2
    """
    global MsunInS, MpcInS, OrbitPeriodInS, OrbitRadiusInS, MearthInS, YearInS, AUInS, Tc, Phic, ThetaS, Phis, Iota, Psi, Z, M1sun, M2sun, DL, Mc, Eta

    MsunInS = 4.926860879228572e-06  # solar mass in [s]
    MpcInS = 102927125053532.6  # mega parsec in [s]
    OrbitRadiusInS = 1e8 / const.c.value  # 1e5 km
    MearthInS = const.M_earth.value * const.G.value / const.c.value**3
    OrbitPeriodInS = 2 * np.pi * np.sqrt(OrbitRadiusInS**3 / MearthInS)
    YearInS = 31536000.0  # one year in [s]
    AUInS = const.au.value / const.c.value  # AU in [s]

    np.random.seed(115719)  # for reproducibility
    data = []  # store generated waveforms
    """
    obtain LISA/Tian Qin noise time series from known PSD
    """
    T = 0.45 * YearInS + 10  # Total observation time
    fs = 0.1  # sampling frequency
    df = 1 / T  # frequency resolution
    fmin = df
    fmax = fs / 2
    N = int(T * fs)  # number of samples
    Nf = int((N - np.mod(N, 2)) / 2)  # number of frequency bins
    f = np.linspace(fmin, fmax, Nf)
    t = np.linspace(0, T, N)

    ################
    # PSD = TQPSD(f)
    PSD = LISAPSD(f)
    ################

    Nt = 10  # number of waveform templates
    for i in range(0, Nt):
        # generate noise time series according to certain PSD
        # TimeSeries, TimeVector = psd2timeseries(PSD, fmax)
        # Noise_t = TimeSeries
        noise = psd2noise(T, fs, N, Nf, PSD)
        # Calculate the PSD
        # freq, Pxx = signal.welch(noise, fs, nperseg=int(N / 3))
        freq, Pxx = signal.periodogram(noise, fs)
        fft_noise = 2 * np.abs(np.fft.fft(noise) / fs) ** 2 / T

        fig, axs = plt.subplots(3, 1, figsize=(5, 10))
        # ax.plot(TimeVector, Noise_t)
        axs[0].plot(t, noise)
        axs[0].set_title("Noise time series n(t)")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Noise n(t)")
        # plot simulated PSD vs. designed PSD
        axs[1].loglog(f, np.sqrt(fft_noise[1 : Nf + 1]), label="FFT PSD")
        axs[1].loglog(f, np.sqrt(PSD), label="Designed PSD")
        axs[1].set_title("PSD")
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("$\sqrt{\mathrm{PSD}}$ [Hz$^{-1/2}$]")
        axs[1].legend()
        # plot periodogram PSD vs. designed PSD
        axs[2].loglog(freq[1:Nf], np.sqrt(Pxx[1:Nf]), label="Periodogram PSD")
        axs[2].loglog(f, np.sqrt(PSD), label="Designed PSD")
        axs[2].set_title("PSD")
        axs[2].set_xlabel("Frequency [Hz]")
        axs[2].set_ylabel("$\sqrt{\mathrm{PSD}}$ [Hz$^{-1/2}$]")
        axs[2].legend()
        fig.tight_layout()
        fig.savefig("./Data/Noise" + str(i) + ".png", dpi=350)

        # setup source parameters
        Tc = np.random.rand(1)[0] * YearInS  # chirp time
        Phic = 0.954  # chirp phase
        # ThetaS = 1.325
        # PhiS = 2.04
        # Iota = 1.02
        # Psi = 0.658
        ThetaS = np.arccos(np.random.uniform(-1, 1))
        PhiS = np.random.uniform(0, 2 * np.pi)
        Iota = np.arccos(np.random.uniform(-1, 1))
        Psi = np.random.uniform(0, 2 * np.pi)
        # print(f"ThetaS is {ThetaS}, PhiS is {PhiS}, Iota is {Iota}, Psi is {Psi}")

        Z = (
            10 * np.random.rand(1)[0]
        )  # choose sources within z < 10 #1.0  # cosmological redshift
        M1sun = 1e7  # solar mass as unit
        M2sun = 1e6  # solar mass as unit
        # Chi1   = 0.1    # dimensionless parameter
        # Chi2   = 0.0    # dimensionless parameter
        cosmo = FlatLambdaCDM(H0=67, Om0=0.32)
        DL = cosmo.luminosity_distance(Z).value * MpcInS  # Mpc in second
        M1 = (1 + Z) * M1sun * MsunInS  # solar mass in second
        M2 = (1 + Z) * M2sun * MsunInS  # solar mass in second
        M = M1 + M2  # total mass
        Qmr = M1 / M2  # mass ratio
        Mu = M1 * M2 / M  # reduced mass
        # TODO: 1. choose Mc randomly in some range. 2. choose Eta randomly in some range.
        Mc = Mu ** (3.0 / 5) * M ** (2.0 / 5)  # chirp mass
        Eta = M1 * M2 / M**2  # symmetric mass ratio
        """
        signal simulation
        """
        # parameter setup
        tc_true = Tc
        phic_true = Phic
        mc_true = Mc
        eta_true = Eta
        dl_true = DL
        thetas_true = ThetaS
        phis_true = PhiS
        iota_true = Iota
        psi_true = Psi

        para = np.array(
            [
                tc_true,
                phic_true,
                mc_true,
                eta_true,
                dl_true,
                thetas_true,
                phis_true,
                iota_true,
                psi_true,
            ]
        )  # convert parameter space to np array

        # print("Observation time: %f year" % Tobs)
        # print("Sampling time interval: %f second" % (TimeVector[2] - TimeVector[1]))
        h_t = ht_model(
            # TimeVector,
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

        # TODO: make white gaussian noise in scale
        Noise_gaus = np.random.randn(1, len(h_t))

        # save Data into pandas dataframe
        # Data = h_t + Noise_gaus
        Data = h_t + noise
        wave_length = len(Data)
        waveform = np.concatenate((Data, para))
        data.append(waveform)

        # Whiten data
        Data_white = whiten_data(Data, PSD, fs)
        # TODO: Calculate SNR

        fig, ax = plt.subplots(2, 1, figsize=(5, 5))
        ax[0].plot(t, h_t, color="r")
        ax[0].set_title("Signal model")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Signal h(t)")
        ax[1].plot(t, Data, color="y", label="Data")
        ax[1].plot(t, h_t, color="r", label="Signal")
        ax[1].plot(t, Data_white, color="b", label="Whitened Data")
        ax[1].set_title("Data model")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Data d(t)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig("./Data/Data_Model" + str(i) + ".png", dpi=350, bbox_inches="tight")

        plt.close("all")  # close all open figs to save memory

    data_length = len(data[0])
    para_names = "tc_true,phic_true,mc_true,eta_true,dl_true,thetas_true,phis_true,iota_true,psi_true".split(
        ","
    )
    columns = list(range(wave_length, data_length))
    columns_str = [str(x) for x in columns]
    # print(columns_str)
    # print(f'data length is {data_length}')
    # print(f'wave length is {wave_length}')
    df = pd.DataFrame(data)
    new_columns = {key: value for key, value in zip(columns_str, para_names)}
    df.rename(columns=new_columns)
    # print(f'New columns is {new_columns}')
    df.to_csv("./Data/waveforms.csv", index=False)
    return 0


# h(t) models for Tian Qin and LISA


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

    # """the frequency-domain amplitude: 1st 60 degree MI with antenna response (contain plus and cross mode) for TianQin"""
    ht_LISA = (
        -mc
        * Q_LISA
        / dl
        * ((tc - t) / (5 * mc)) ** (-1.0 / 4)
        * np.cos(phip_LISA + 2 * PSI_PN)
    )
    return ht_LISA


# h(t) cut down at Tc

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


# get noise(t) from a known PSD


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


# ref：https://groups.google.com/g/comp.soft-sys.matlab/c/tmw2H26MDtI


### ref：https://groups.google.com/g/comp.soft-sys.matlab/c/tmw2H26MDtI
def psd2timeseries(PSD, fmax):
    """
    x = time series
    t = time vector
    PSD = power spectral density (one-sided, from 0 to fNyq)
    fmax = max frequecy can reach if fmax < Nyquist frequency
    """

    N = len(PSD)

    # Sampling of frequency vector
    delta_f = fmax / (N - 1)

    # Compute amplitude of frequency spectrum
    # multiply PSD by fs*N=2*fNyq*N (a factor to match)
    SpectrumAmplitude = np.sqrt(PSD * 2 * fmax * N)

    # Compute phase of frequency spectrum
    SpectrumPhase = np.random.rand(len(PSD)) * 2 * np.pi

    # Compute complex spectrum
    Spectrum = SpectrumAmplitude * np.exp(1j * SpectrumPhase)
    Spectrum_n = np.conj(
        Spectrum[1:][::-1]
    )  # reverse and conjugate and exclude the DC part

    # Generate the two-sided PSD:
    Spectrum_TwoSided = np.concatenate([Spectrum, Spectrum_n])
    # print(f'length of PSD {len(PSD)}\n length of Spectrum {len(Spectrum)}\n length of Spectrum_TwoSided {len(Spectrum_TwoSided)}')

    # Compute inverse FFT
    x = ifft(Spectrum_TwoSided)
    #     x= fftshift(x)
    x = np.real(x)

    # Compute time vector
    delta_t = 1.0 / (2.0 * fmax)
    t = np.arange(0, (N * 2 - 1) * delta_t, delta_t)

    return [x, t]


def psd2noise(T, fs, N, Nf, PSD):
    """
    Generate noise time series from a given PSD.

    Parameters:
    T: total observation time
    fs: sampling frequency
    N: number of samples
    Nf: number of frequency bins
    PSD: given one-sided power spectral density
    """
    w_noise = np.random.randn(N)
    xf_white = np.fft.fft(w_noise)
    # concatenate one-sdided PSD into tow-sided
    PSD = np.concatenate((PSD, PSD[-1 - np.mod(N + 1, 2) :: -1]))
    # generate colored noise
    xf_noise = xf_white[1:] * np.sqrt(PSD * fs)
    xf_noise = np.insert(xf_noise, 0, 0)  # insert DC component
    x = np.real(np.fft.ifft(xf_noise))
    return x


def whiten_data(data, psd, fs):
    """
    Whiten data given a psd and sample rate

    Parameters:
    data: data to be whitened
    psd: given one-sided power spectral density
    fs: sample rate
    """
    N = len(data)
    # calculate the fft of data
    xf = np.fft.fft(data)
    # concatenate one-sdided PSD into tow-sided
    psd = np.concatenate((psd, psd[-1 - np.mod(N + 1, 2) :: -1]))
    # whiten the data
    xf_white = xf[1:] / np.sqrt(1 / 2 * psd)
    xf_white = np.insert(xf_white, 0, 0)  # add DC component
    # recover the time series
    x = np.real(np.fft.ifft(xf_white))
    return x


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
