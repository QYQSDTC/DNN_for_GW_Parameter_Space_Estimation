# DNN for GW Hyperarameter Estimation
Using Depp Learning method to estimate GW parameter space by training DNN with a GW template bank.

Trying to extend the job done in reference.

## Data Preparation
1. Generate pure LISA signals as templates(~ 3000) 
2. obtain real noise time series from LISA noise PSD(Power Spectral Density)
3. Whiten our data using LISA PSD before add multiple realizations of :
    1. White Gaussian Noise
    2. Real LISA Noise
4. Shift signal peak to left at some random time
5. Add multiple noise realizations to shifted(whitened(signal))

## DNN Design
We use, at first, the same DNN used in **Reference**.

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
