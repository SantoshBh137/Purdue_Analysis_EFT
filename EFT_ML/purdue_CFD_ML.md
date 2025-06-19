# EFT vs SM Overview

This repository explores the distinction between **Standard Model (SM)** and **Effective Field Theory (EFT)** predictions for top quark pair production, with a focus on sensitivity to the `ctGRe` Wilson coefficient.

We use Monte Carlo simulated ROOT files as well as preprocessed `.pkl` files to build classification models and visualize EFT-sensitive observables.

---

## Data Sources

### ROOT Files

We begin with two ROOT files representing different physical hypotheses:

- `eft_ctgRe0.root` — **Standard Model (SM)** prediction (`ctGRe = 0`)
- `eft_ctgRe2.root` — **Effective Field Theory (EFT)** prediction (`ctGRe = 2`)

Each file contains two main branches:

| Branch                      | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `ttBar_treeVariables_step0` | Generator-level (**GEN**) observables, unaffected by detector simulation   |
| `ttBar_treeVariables_step8` | Reconstructed-level (**RECO**) observables, including detector effects      |

These branches include:

- Kinematic observables (`pt`, `eta`, `phi`, `mass`, etc.)
- Spin correlation and polarization observables(`b1k`,`b2k`..,`c_kk`,`c_rr`,`c_nn`,...)
- Event weights, especially `finalWeight`

> **Note:** The physical effect of SMEFT operators is **entirely encoded in the `finalWeight`**. Without applying it, EFT and SM samples will appear nearly identical.

---

###  Preprocessed `.pkl` Files (Recommended)

To simplify analysis, we provide the following preprocessed pickle files:

- `sample_SM_ctgRe0.pkl` — SM sample
- `sample_EFT_ctgRe2.pkl` — EFT sample with `ctGRe = 2`

These `.pkl` files include:

- **GEN** and **RECO** observables as separate columns
- The effect of the `finalWeight` already applied

> ⚠️ **You do not need to apply any weight manually.** These files are ready to use directly for ML training and plotting.




---
## Objective

The primary goal of this analysis is to perform **classification between SM and EFT** predictions, using observables either at the **GEN** or **RECO** level.

* **GEN-level classification** is expected to be easier due to the absence of detector smearing.
* **RECO-level classification** is more realistic but may be more challenging due to detector effects.

I have also plotted several observables that are particularly sensitive to the `ctGRe` Wilson coefficient. You can explore these plots in the following notebook:
 [`Plotting_pkl.ipynb`](https://github.com/SantoshBh137/Purdue_Analysis_EFT/blob/main/EFT_ML/Plotting_pkl.ipynb)

In addition, I have implemented a simple deep neural network model to classify SM and EFT samples. The training and evaluation of the model are available here:
 [ML\_EFT\_SM.ipynb](https://github.com/SantoshBh137/Purdue_Analysis_EFT/blob/main/EFT_ML/ML_EFT_SM.ipynb)

---

## Likelihood Approximation in EFT

An important task in EFT analysis is to **approximate the likelihood function**, which enables hypothesis testing and parameter inference.

For a comprehensive guide on likelihood-based approaches using machine learning in the EFT context, refer to this Master's thesis:
[Jonas Rübenach – Likelihood approximation using ML in EFT](https://bib-pubdb1.desy.de/record/425819/files/Jonas%20R%C3%BCbenach%20Master%20thesis.pdf)

Also, this one [Constraining Effective Field Theories with Machine Learning] https://arxiv.org/pdf/1805.00013

