import numpy as np
from functools import partial
from astropy import constants as const
from astropy import units as u

YearInS = (1 * u.yr).to(u.s).value  # one year in [s]
OrbitRadiusInS = 1e8 / const.c.value  # 1e5 km
MearthInS = const.M_earth.value * const.G.value / const.c.value**3
OrbitPeriodInS = 2 * np.pi * np.sqrt(OrbitRadiusInS**3 / MearthInS)

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