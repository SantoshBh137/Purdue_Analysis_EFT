# Introduction

One of the main goals at the LHC is to search for signs of new physics beyond the Standard Model (SM). However, new particles may be too heavy to be produced directly at the energies accessible by the LHC. Instead, they can **subtly alter the interactions** of known particles.

Think of it like someone stomping on the floor above you—you don’t see them, but you feel the vibrations. Similarly, **heavy new physics leaves subtle “ripples”** in observable data, and our job is to detect those ripples.

To describe such indirect effects in a model-independent way, we use **Effective Field Theory (EFT)**. EFT allows us to express the influence of unknown high-energy physics using known particles and new interaction terms called **higher-dimensional operators**.

These operators are organized by how suppressed they are by a large energy scale \( \Lambda \), where new physics “lives.” The effects appear in an expansion like:

\[
\mathcal{L}_\text{EFT} = \mathcal{L}_\text{SM} + \frac{1}{\Lambda^2} \sum_i f_i \mathcal{O}_i + \dots
\]

- \( \mathcal{O}_i \): dimension-six operators  
- \( f_i \): Wilson coefficients  
- \( \Lambda \): the energy scale of new physics  
- \( \mathcal{L}_\text{SM} \): Standard Model Lagrangian

Since these effects are suppressed by \( 1/\Lambda^2 \), they become more pronounced as the ratio \( E/\Lambda \) (experiment energy over new physics scale) increases. The **smaller** this ratio, the **more reliable** EFT becomes.

**Analogy:** Imagine a rock thrown into a lake:
- The farther you are (larger \( \Lambda \)), the smaller the ripples you feel.
- The closer you are (larger \( E/\Lambda \)), the more the ripples affect you—and the more you must account for higher-order terms.

---

## Why EFT Inference Is Hard

EFT measurements at the LHC face three key challenges:

1. **High-dimensional parameter space**  
   Many EFT operators can affect a single process. Each operator may influence multiple observables, and we must account for correlations between them.

2. **Complex event structure**  
   The effects of EFTs appear in subtle patterns across many observables. Traditional analyses using just a few variables can miss important signals.

3. **Intractable likelihood**  
   Simulators like **Pythia** and **Geant4** are used to model particle collisions, parton showers, and detector responses. These tools can generate synthetic events but **do not provide an explicit formula** for the likelihood \( p(x|\theta) \) of observing a specific event \( x \) under theory \( \theta \).

### Why Not?

Each simulation goes through multiple stochastic stages:

| Stage                | Tool               | Description                                      |
|---------------------|--------------------|--------------------------------------------------|
| Hard scattering      | MadGraph, Pythia   | Generates the parton-level interaction           |
| Parton shower        | Pythia             | Simulates QCD radiation (quark/gluon branching)  |
| Hadronization        | Pythia             | Converts partons into hadrons                    |
| Detector simulation  | Geant4             | Models particle interactions with the detector   |
| Reconstruction       | CMS software       | Reconstructs physical observables (jets, MET)    |

Each stage involves:
- Random processes,
- Millions of hidden variables (e.g., decay paths, detector noise),
- Is **not invertible** — you can’t deterministically trace an observed event back to the original theory.

Mathematically, the likelihood is:

\[
p(x|\theta) = \int p(x|z) \cdot p(z|\theta) \, dz
\]

Where:
- \( z \): parton-level variables (latent),
- \( x \): observed event,
- \( \theta \): EFT parameters.

This integral is **computationally intractable** because it sums over a vast space of latent variables.

### Analogy: The Maze

Imagine a giant maze with many random paths:
- You can walk through the maze many times and record which exits (events) you reach.
- But if someone shows you a specific exit and asks:
  > “What’s the probability someone came out here?”

  You’d need to **sum over all possible paths** that could have led to that exit — too many to track.

---

## Why Traditional Methods Fall Short

One common workaround is to select a **small number of discriminating variables** (e.g. invariant mass or transverse momentum) and fill **histograms** from simulations to compare different hypotheses. This is the basis of **template fits**.

While these methods are simple and interpretable, they have serious limitations:
- They **ignore most of the available data**, using only 1–2 variables out of dozens,
- They **miss correlations** between observables,
- They **fail to capture subtle, multidimensional patterns** introduced by EFT operators.

This often leads to **poor sensitivity** in parts of the EFT parameter space where the effects are spread across many observables.

---

## A Machine Learning-Based Solution

To get around the intractable likelihood and the limitations of histograms, physicists and ML researchers are turning to **machine learning techniques** that:

- Use simulated samples from different EFT parameter settings,
- Learn patterns and differences in high-dimensional feature space,
- Estimate useful quantities like the **likelihood ratio** directly.

### ML Classifiers for Likelihood-Free Inference

Here’s how it works in practice:

1. Generate two sets of simulated events:
   - One with EFT parameters \( \theta_0 \),
   - Another with EFT parameters \( \theta_1 \).

2. Train a **classifier** (like a neural network) to distinguish between the two.

3. The classifier output can be transformed into an **approximate likelihood ratio**:

\[
\hat{r}(x \mid \theta_0, \theta_1) \approx \frac{p(x \mid \theta_1)}{p(x \mid \theta_0)}
\]

This method:
- Doesn’t require the likelihood to be known,
- Works in **high-dimensional spaces**,
- **Automatically captures correlations** between observables,
- Provides a statistically optimal test statistic for comparing models.

---

This machine learning approach opens the door to **precise and scalable EFT measurements**, even in the face of complex simulations and high-dimensional observables.
