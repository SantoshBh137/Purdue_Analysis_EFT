## Overview of EFT vs SM ROOT Files

This repository works primarily with two ROOT files:

* **`eft_ctgRe0.root`** – Represents the **Standard Model (SM)** prediction. This file contains only operators up to dimension 4, meaning no new physics effects are included.
* **`eft_ctgRe2.root`** – Represents an **Effective Field Theory (EFT)** prediction that includes **dimension-6 operators**, specifically those coupled to the **`ctGRe` Wilson coefficient**, indicating the presence of potential new physics effects.

For a detailed theoretical background on EFT predictions, see [arXiv:2406.14620](https://arxiv.org/pdf/2406.14620).

---

## Structure of the ROOT Files

Each ROOT file contains two main branches:

* **`ttBar_treeVariables_step0`**:
  Contains **generator-level (GEN)** or **truth-level** information. These observables are unaffected by detector simulation, providing a cleaner view of the theoretical predictions.

* **`ttBar_treeVariables_step8`**:
  Contains **reconstructed-level (RECO)** information. This includes the same physics observables as `step0`, but with detector effects taken into account.

Each branch contains a variety of physics observables:

* **Kinematic variables** such as `pt`, `eta`, `phi`, `mass`
* **Polarization and spin correlation observables**
* **Weights**, specifically `finalWeight`

The **`finalWeight`** branch plays a crucial role in distinguishing SM from EFT predictions. The effect of new physics (EFT) or the Standard Model only becomes visible when the observables are used **in combination with these weights**.

> ⚠️ If you plot the observables from `eft_ctgRe0.root` and `eft_ctgRe2.root` without applying the `finalWeight`, the distributions will look identical. This is because the physical effect of the SMEFT operators is **encoded entirely in the weights**.

Please keep this in mind when analyzing or comparing the files.

For more information about spin correlations and polarization observables, refer to [arXiv:1907.03729](https://arxiv.org/pdf/1907.03729).

---

## Objective

The primary goal of this analysis is to perform **classification between SM and EFT** predictions, using observables either at the **GEN** or **RECO** level.

* **GEN-level classification** is expected to be easier due to the absence of detector smearing.
* **RECO-level classification** is more realistic but may be more challenging due to detector effects.

I have also plotted several observables that are particularly sensitive to the `ctGRe` Wilson coefficient. You can explore these plots in the following notebook:
🔗 [Observables\_plotting.ipynb](https://github.com/SantoshBh137/Purdue_Analysis_EFT/blob/main/EFT_ML/Observables_plotting.ipynb)

In addition, I have implemented a simple deep neural network model to classify SM and EFT samples. The training and evaluation of the model are available here:
🔗 [ML\_EFT\_SM.ipynb](https://github.com/SantoshBh137/Purdue_Analysis_EFT/blob/main/EFT_ML/ML_EFT_SM.ipynb)

---

## Likelihood Approximation in EFT

An important task in EFT analysis is to **approximate the likelihood function**, which enables hypothesis testing and parameter inference.

For a comprehensive guide on likelihood-based approaches using machine learning in the EFT context, refer to this Master's thesis:
[Jonas Rübenach – Likelihood approximation using ML in EFT](https://bib-pubdb1.desy.de/record/425819/files/Jonas%20R%C3%BCbenach%20Master%20thesis.pdf)
