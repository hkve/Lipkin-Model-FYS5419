# Lipkin-Model-FYS5419

This repository contains code developed during the semester project of FYS5419 – Quantum computing and quantum machine learning. The code has been written by [
Bendik Selvaag-Hagen](https://github.com/bendiksh96) and [Håkon Kvernmoen](https://github.com/hkve/). We used the Variational Eigen Quantomsolver (VQE) to calculate ground state energies of different systems, ranging from one to four qbit circuits. In particular, the Lipkin Model has been considered for both two and four particles and compared with Full Configuration Interaction (FCI), Hartree-Fock (HF) and Random Phase Approximation (RPA) calculations.

The directories `src` and `notebooks` contain the code used to calculate energies and create figures, while `src` contains our associated article. Running these should be quite straightforward. For full usage, the following packages should be installed

- `numpy>=1.21.5`
- `qiskit>=0.43.1`
- `matplotlib>=3.5.1`
- `scipy>=1.8.0`
- `seaborn>=0.12.2`

