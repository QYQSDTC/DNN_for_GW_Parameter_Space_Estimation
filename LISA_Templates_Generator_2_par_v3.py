#!/home/ud202180035/miniconda3/envs/dnn/bin/python
"""
This script generates waveform templates for LISA.
Modified from WenFan Feng's code.
Aknowledge Yao Fu's helpful discussion.

v2: Each process writes a single HDF5 file, with each waveform as a separate group.
v3: Adds a second injected signal h_t2 per waveform. Two independent parameter sets
    are sampled from the same config2.yaml ranges; both signals are rescaled to the
    same target SNR and superimposed on a single shared noise realization:
        Data = h_t + h_t2 + noise
    The HDF5 group stores white_signal, white_signal2 and both *_true / *_true2 attrs.
    Outputs are written to waveforms2_2sig/, fig2_2sig/ and sim2_2sig.log so that
    the v2 outputs are not overwritten.
"""

import os
import time
from multiprocessing import Pool

import matplotlib as mpl
import numpy as np

from LISA_TianQin_Waveforms import *

# 设定agg.path.chunksize值，例如设置为10000
mpl.use("Agg")
mpl.rcParams["agg.path.chunksize"] = 10000
import h5py
import matplotlib.pyplot as plt
import yaml
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.fftpack import fft, ifft
from scipy.stats import loguniform

# read parameters from config file
with open("config2.yaml", "r") as f:
    config = yaml.safe_load(f)
# setup units and constants
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


def main():
    Tc_min, Tc_max = config["Tc"]  # chirp time
    ThetaS_min, ThetaS_max = config["ThetaS"]  # sky location
    PhiS_min, PhiS_max = config["PhiS"]  # sky location
    Iota_min, Iota_max = config["Iota"]  # inclination
    Phic_min, Phic_max = config["Phic"]  # chirp phase
    Psi_min, Psi_max = config["Psi"]  # polarization

    Z = 10  # cosmological redshift
    M1sun_min, M1sun_max = config["M1sun"]  # solar mass as unit
    M2sun_min, M2sun_max = config["M2sun"]  # solar mass as unit
    cosmo = FlatLambdaCDM(H0=67, Om0=0.32)
    DL = cosmo.luminosity_distance(Z).value * MpcInS  # Mpc in second
    snr = config["SNR"]
    n_processes = config.get("n_processes", 4)
    n_waveforms = config["n_waveforms"]
    n_per_process = n_waveforms // n_processes

    pool = Pool(processes=n_processes)
    args = [
        (
            T,
            t,
            f,
            fs,
            N,
            Phic_min,
            Phic_max,
            DL,
            Iota_min,
            Iota_max,
            Psi_min,
            Psi_max,
            Tc_min,
            Tc_max,
            M1sun_min,
            M1sun_max,
            M2sun_min,
            M2sun_max,
            ThetaS_min,
            ThetaS_max,
            PhiS_min,
            PhiS_max,
            snr,
            Z,
            i,  # process index
            i * n_per_process,  # cnt offset for random seed
            n_per_process,
        )
        for i in range(n_processes)
    ]

    pool.map(process_M1_M2, args)
    pool.close()
    pool.join()
    print("All subprocesses done.")


# process M1 and M2
def process_M1_M2(args):
    """
    Process M1 and M2 for parallelization.
    Each process writes all its waveforms into a single HDF5 file.
    Two independent parameter sets are sampled per waveform to build h_t and h_t2.
    """
    (
        T,
        t,
        f,
        fs,
        N,
        Phic_min,
        Phic_max,
        DL,
        Iota_min,
        Iota_max,
        Psi_min,
        Psi_max,
        Tc_min,
        Tc_max,
        M1sun_min,
        M1sun_max,
        M2sun_min,
        M2sun_max,
        ThetaS_min,
        ThetaS_max,
        PhiS_min,
        PhiS_max,
        snr,
        Z,
        proc_idx,
        cnt_offset,
        n_waveforms_per_process,
    ) = args

    # set different random seeds for different processors
    np.random.seed(cnt_offset)

    os.makedirs("waveforms2_2sig", exist_ok=True)
    os.makedirs("fig2_2sig", exist_ok=True)

    # Open a single HDF5 file per process
    h5_path = f"waveforms2_2sig/waveforms_proc{proc_idx}.h5"
    with h5py.File(h5_path, "w") as h5file:
        cnt = cnt_offset
        for _ in range(n_waveforms_per_process):
            # ---- Sample params for the first signal h_t ----
            M1sun = loguniform.rvs(M1sun_min, M1sun_max)
            M2sun = loguniform.rvs(M2sun_min, M2sun_max)
            M1 = (1 + Z) * M1sun * MsunInS
            M2 = (1 + Z) * M2sun * MsunInS
            M = M1 + M2
            Mu = M1 * M2 / M
            Mc = Mu ** (3.0 / 5) * M ** (2.0 / 5)
            Eta = M1 * M2 / M**2

            Tc = np.random.uniform(Tc_min, Tc_max) * YearInS
            ThetaS = np.arccos(np.random.uniform(ThetaS_min, ThetaS_max))
            PhiS = np.random.uniform(PhiS_min, PhiS_max)
            Iota = np.arccos(np.random.uniform(Iota_min, Iota_max))
            Psi = np.random.uniform(Psi_min, Psi_max)
            Phic = np.random.uniform(Phic_min, Phic_max)

            # ---- Independently sample params for the second signal h_t2 ----
            M1sun2 = loguniform.rvs(M1sun_min, M1sun_max)
            M2sun2 = loguniform.rvs(M2sun_min, M2sun_max)
            M1_2 = (1 + Z) * M1sun2 * MsunInS
            M2_2 = (1 + Z) * M2sun2 * MsunInS
            M_2 = M1_2 + M2_2
            Mu_2 = M1_2 * M2_2 / M_2
            Mc2 = Mu_2 ** (3.0 / 5) * M_2 ** (2.0 / 5)
            Eta2 = M1_2 * M2_2 / M_2**2

            Tc2 = np.random.uniform(Tc_min, Tc_max) * YearInS
            ThetaS2 = np.arccos(np.random.uniform(ThetaS_min, ThetaS_max))
            PhiS2 = np.random.uniform(PhiS_min, PhiS_max)
            Iota2 = np.arccos(np.random.uniform(Iota_min, Iota_max))
            Psi2 = np.random.uniform(Psi_min, Psi_max)
            Phic2 = np.random.uniform(Phic_min, Phic_max)

            for snr_ in snr:
                generate_waveform(
                    T,
                    t,
                    f,
                    fs,
                    N,
                    Tc,
                    Phic,
                    Mc,
                    Eta,
                    DL,
                    ThetaS,
                    PhiS,
                    Iota,
                    Psi,
                    Tc2,
                    Phic2,
                    Mc2,
                    Eta2,
                    DL,
                    ThetaS2,
                    PhiS2,
                    Iota2,
                    Psi2,
                    snr_,
                    cnt,
                    h5file,
                )
                with open("sim2_2sig.log", "a") as log_file:
                    log_file.write(
                        f"waveform{cnt}_SNR{snr_:.2f}: "
                        f"Tc: {Tc}, Mc: {Mc}, SNR: {snr_}, Phic: {Phic}, Eta: {Eta}, "
                        f"DL: {DL}, ThetaS: {ThetaS}, PhiS: {PhiS}, Iota: {Iota}, Psi: {Psi}, "
                        f"Tc2: {Tc2}, Mc2: {Mc2}, Phic2: {Phic2}, Eta2: {Eta2}, "
                        f"ThetaS2: {ThetaS2}, PhiS2: {PhiS2}, Iota2: {Iota2}, Psi2: {Psi2}\n"
                    )
            cnt += 1

    print(f"Process {proc_idx} done. Saved {n_waveforms_per_process} waveforms to {h5_path}")


# generate waveform
def generate_waveform(
    T,
    t,
    f,
    fs,
    N,
    tc_true,
    phic_true,
    mc_true,
    eta_true,
    dl_true,
    thetas_true,
    phis_true,
    iota_true,
    psi_true,
    tc_true2,
    phic_true2,
    mc_true2,
    eta_true2,
    dl_true2,
    thetas_true2,
    phis_true2,
    iota_true2,
    psi_true2,
    snr,
    cnt,
    h5file,
):
    """
    Generate a waveform containing two independent signals h_t and h_t2 with given
    parameters and write to an open HDF5 file.

    Args:
        T: total time
        t: time vector
        f: frequency vector
        fs: sampling frequency
        N: number of data length
        tc_true, phic_true, mc_true, eta_true, dl_true,
        thetas_true, phis_true, iota_true, psi_true:
            parameters of the first signal h_t
        tc_true2, phic_true2, mc_true2, eta_true2, dl_true2,
        thetas_true2, phis_true2, iota_true2, psi_true2:
            parameters of the second signal h_t2 (independently sampled)
        snr: target signal-to-noise ratio applied to both signals separately
        cnt: waveform number
        h5file: open HDF5 file handle
    """
    # calculate PSD and convert it to two-sided PSD
    PSD = LISAPSD(f)
    PSD[0] = 0  # set the DC part to zero
    PSD2 = (
        np.concatenate([PSD, np.conj(PSD[N % 2 - 2 : 0 : -1])]) / 2
    )  # double-side PSD

    print(
        f"length of PSD {len(PSD)}, length of two-sided {len(PSD2)}, total number of points {N}"
    )

    # generate a white noise realization (shared between the two signals)
    w_noise = np.random.randn(len(PSD2)) * np.sqrt(fs)
    # calculate the noise spectrum
    fft_w_noise = np.fft.fft(w_noise)
    # multiply the spectrum by the square root of the PSD
    fft_noise = fft_w_noise * np.sqrt(PSD2)
    # calculate the inverse FFT
    noise = np.real(np.fft.ifft(fft_noise))

    # ===== First signal h_t =====
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
    snr_lisa = SNR(data=h_t, T=T, fs=fs, detector="LISA")
    print(f"SNR of signal 1: {snr_lisa:.2f}")
    print(f"Desired SNR is {snr}")
    scale = snr / snr_lisa
    dl_true_rs = dl_true / scale
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
    snr_lisa = SNR(data=h_t, T=T, fs=fs, detector="LISA")
    print(f"Re-scaled SNR of signal 1: {snr_lisa:.2f}")

    # ===== Second signal h_t2 =====
    h_t2 = ht_model(
        t,
        tc_true2,
        phic_true2,
        mc_true2,
        eta_true2,
        dl_true2,
        thetas_true2,
        phis_true2,
        iota_true2,
        psi_true2,
    )
    snr_lisa2 = SNR(data=h_t2, T=T, fs=fs, detector="LISA")
    print(f"SNR of signal 2: {snr_lisa2:.2f}")
    scale2 = snr / snr_lisa2
    dl_true2_rs = dl_true2 / scale2
    h_t2 = ht_model(
        t,
        tc_true2,
        phic_true2,
        mc_true2,
        eta_true2,
        dl_true2_rs,
        thetas_true2,
        phis_true2,
        iota_true2,
        psi_true2,
    )
    snr_lisa2 = SNR(data=h_t2, T=T, fs=fs, detector="LISA")
    print(f"Re-scaled SNR of signal 2: {snr_lisa2:.2f}")

    # Two signals share a single noise realization
    Data = h_t + h_t2 + noise

    # whiten Data
    fft_Data = np.fft.fft(Data)
    white_Data = fft_Data / np.sqrt(PSD2)
    # set the first and last element to zero
    white_Data[0] = 0
    White_Data_t = np.real(np.fft.ifft(white_Data))
    # drop the first and last 1% of the data
    White_Data_t = White_Data_t[int(0.02 * N) : int(0.99 * N)]
    t_plot = t[int(0.02 * N) : int(0.99 * N)]

    # whiten the signal h_t
    fft_h_t = np.fft.fft(h_t)
    white_h_t = fft_h_t / np.sqrt(PSD2)
    white_h_t[0] = 0
    White_h_t = np.real(np.fft.ifft(white_h_t))
    White_h_t = White_h_t[int(0.02 * N) : int(0.99 * N)]

    # whiten the signal h_t2
    fft_h_t2 = np.fft.fft(h_t2)
    white_h_t2 = fft_h_t2 / np.sqrt(PSD2)
    white_h_t2[0] = 0
    White_h_t2 = np.real(np.fft.ifft(white_h_t2))
    White_h_t2 = White_h_t2[int(0.02 * N) : int(0.99 * N)]

    # Save white_Data_t and parameters as a group in the shared HDF5 file
    grp = h5file.create_group(f"waveform{cnt}_SNR{snr:.2f}")
    grp.create_dataset("white_Data", data=White_Data_t)
    grp.create_dataset("white_signal", data=White_h_t)
    grp.create_dataset("white_signal2", data=White_h_t2)
    # First-signal parameters
    grp.attrs["tc_true"] = tc_true
    grp.attrs["phic_true"] = phic_true
    grp.attrs["mc_true"] = mc_true
    grp.attrs["eta_true"] = eta_true
    grp.attrs["dl_true"] = dl_true
    grp.attrs["thetas_true"] = thetas_true
    grp.attrs["phis_true"] = phis_true
    grp.attrs["iota_true"] = iota_true
    grp.attrs["psi_true"] = psi_true
    # Second-signal parameters
    grp.attrs["tc_true2"] = tc_true2
    grp.attrs["phic_true2"] = phic_true2
    grp.attrs["mc_true2"] = mc_true2
    grp.attrs["eta_true2"] = eta_true2
    grp.attrs["dl_true2"] = dl_true2
    grp.attrs["thetas_true2"] = thetas_true2
    grp.attrs["phis_true2"] = phis_true2
    grp.attrs["iota_true2"] = iota_true2
    grp.attrs["psi_true2"] = psi_true2
    grp.attrs["snr"] = snr

    # Plot Data with both signals, then each pure signal
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), dpi=300)
    axs[0].plot(t_plot, White_Data_t, label="Data")
    axs[0].plot(t_plot, White_h_t, label="signal 1", alpha=0.8)
    axs[0].plot(t_plot, White_h_t2, label="signal 2", alpha=0.8)
    axs[0].set_title(
        "Whitened Data & Signals, SNR1 = %.2f, SNR2 = %.2f" % (snr_lisa, snr_lisa2)
    )
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    # plot pure signal 1
    axs[1].plot(t_plot, White_h_t, label="signal 1")
    axs[1].set_title("Pure Signal 1")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    # plot pure signal 2
    axs[2].plot(t_plot, White_h_t2, label="signal 2", color="tab:green")
    axs[2].set_title("Pure Signal 2")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(
        f"fig2_2sig/waveform{cnt}_SNR{snr:.2f}.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Process Done in {time.time()-start_time:.2f}s")
