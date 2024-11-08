<!--
 * @Author: Yiqian Qian
 * @Description: file content
 * @Date: 2023-09-15 11:13:11
 * @LastEditors: Yiqian Qian
 * @LastEditTime: 2023-09-15 11:38:44
 * @FilePath: /DNN_for_GW_Parameter_Space_Estimation/README.md
-->
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

- M1sun: the mass of the primary blackhole, in the unit of solar mass, M1sun > M2sun. $10^6 M_\odot$ < M1sun < $10^9 M_\odot$.

- M2sun: the mass of the minor blackhole.

here are all the parameters you can play with, change the parameters and make your own waveforms.

## LISA_Templates_Generator
### Dependencies
AstroPy, SciPy, Pandas, Numpy, PyYaml,Matplolib
### Config
Use `config.yaml` to set parameter configration

#### Config 1
The first objective of this project is to predict the merger time *Tc*, we set the data length to 3 month and the merger time is between 3 month and 6 month. We will try to predict the *Tc* with the 3 month long data before merging.
#### Config 2
The second objective is to estimate the key parameters of the GW which are the chirp mass ($\mathcal{M}_c$) and sky locations $\alpha$ and $\delta$. For this case, we set the data length to 2 weeks and the merging happens in this window.

### Runtime
After setting up paramters in `config.yaml`, just use python to run the script `python LISA_Templates_Generator.py`, it will create a folder named `fig` to save the generated figures.

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
