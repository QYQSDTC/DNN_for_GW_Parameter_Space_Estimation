# DNN for GW Hyperarameter Estimation

Using Depp Learning method to estimate GW parameter space by training DNN with a GW template bank.

Trying to extend the job done in **Reference** to hyperparameter space.

## Data Preparation

1. Generate SMBHB GW signals as templates(~ 3000)
2. obtain real noise time series from LISA noise PSD(Power Spectral Density)
3. Whiten our data using LISA PSD before add multiple realizations of :
   1. White Gaussian Noise
   2. Real LISA Noise
4. Shift signal peak to left at some random time
5. Add multiple noise realizations to shifted(whitened(signal))

## DNN Design

We use, at first, the same DNN used in **Reference**.

# Usage

## GW_Waveform_Generator

This script can quickly generate GW waveforms for LISA and TianQin using known noise PSD, this section briefly introduces the meaning of each parameter used in the script.

- N: number of templates intend to generate.

- Tc: the time when the signal peaks, can be taken within [0, Tobs].

- Tobs: time of the observation in the unit of year.

- Phic: the chirp phase, in the range of [0, $2\pi$].

- ThetaS: an angle in the range of [0, 2$\pi$].

- PhiS: an angle in the range of [0, 2$\pi$].

- Iota: an angle in the range of [0, 2$\pi$].

- Psi: an angle in the range of [0, 2$\pi$].

- Z: redshift, in the range of [0, 10].

- DL: distance from the source, calculated according to the specific cosmology model and redshift **Z**.

- M1sun: the mass of the primary blackhole, in the unit of solar mass, M1sun > M2sun.

- M2sun: the mass of the minor blackhole.

here are all the parameters you can play with, change the parameters and make your own waveforms.

# Reference

```
@article{PhysRevD.97.044039,
  title = {Deep neural networks to enable real-time multimessenger astrophysics},
  author = {George, Daniel and Huerta, E. A.},
  journal = {Phys. Rev. D},
  volume = {97},
  issue = {4},
  pages = {044039},
  numpages = {23},
  year = {2018},
  month = {Feb},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.97.044039},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.97.044039}
}
```
