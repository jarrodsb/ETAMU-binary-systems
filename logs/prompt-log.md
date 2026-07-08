# Prompt Log
---

## Machine learning for planetary orbit stability prediction

Date:
- 2026-07-07

Tool used:
- Claude

Purpose:
- Summarize the SPOCK project, suggest a machine learning framework for orbital stability classification, and lay out the initial steps.

Prompt text:
```text
As a physics graduate student, I want to use machine learning to predict the stability of planetary orbits in binary star systems.
The package that I seek to use as a template (either directly or inspirationally) is called SPOCK, whose primary features are summarized
 in the README section of the following GitHub page: https://github.com/dtamayo/spock. The core functionality of SPOCK is orbital stability
 prediction of planets, which is described in the paper readable at the following link: https://www.pnas.org/doi/full/10.1073/pnas.2001258117.
 The three-body stability problem for a planet in a binary star system is described in the following GitHub documentation page:
 https://saturnaxis.github.io/exoplanet-binary/Tutorials/three-body-stability.html. Summarize the information provided in the given URL sources.
 Suggest one or two distinct machine learning frameworks that are worth exploring and would likely be the most useful in implementing my own
 project, the code for which would be developed for public use. Justify your suggestions. Finally, list the next steps I should take.

```

Input or sources provided:
- https://github.com/dtamayo/spock
- https://www.pnas.org/doi/full/10.1073/pnas.2001258117
- https://saturnaxis.github.io/exoplanet-binary/Tutorials/three-body-stability.html

Output used:
- 

What worked well:
- Provided a concise summary of the relevant topics associated with the machine learning project I seek to implement, and suggested next steps.

What did not work:
- Claude may have failed to access the full text of the Tamayo et al. 2020 paper at pnas.org, and may have relied solely on the abstract of the paper available on arXiv.
- The suggested next steps were a little too broad in scope for a relative noivce to machine learning and full stack development. They covered the entire process to develop a working prototype, rather than the initial steps to get started.

Reusable version:
```text
## Summary of the Three Sources

**1. SPOCK (GitHub README)**

SPOCK — Stability of Planetary Orbital Configurations Klassifier — is a package to determine whether planetary orbital configurations will live long and prosper, incorporating several machine learning and analytical tools for estimating the stability of compact planetary configurations, all sharing a common API to facilitate comparisons between them and with N-body integrations. It bundles several distinct models built up across multiple papers:

- **FeatureClassifier** (Tamayo et al. 2020) — an XGBoost-based classifier returning a scalar stability probability.
- **DeepRegressor** (Cranmer et al. 2021) — a deep-learning model estimating the *median expected instability time* with uncertainty samples.
- **AnalyticalClassifier** (Tamayo et al. 2021) — a semi-analytic criterion for whether a configuration is dynamically chaotic (regular vs. chaotic).
- **CollisionMergerClassifier / CollisionOrbitalOutcomeRegressor** (Lammers et al. 2024) — MLP-based models predicting which planets collide and the resulting post-collision orbital state.
- **GiantImpactPhaseEmulator** — chains the above together to emulate a full giant-impact phase until long-term stability is reached.

Dependencies include `rebound` (N-body), `xgboost`, `pytorch`/`pytorch-lightning`, and standard scikit-learn tooling — a useful signal for what your own stack might look like.

**2. Tamayo et al. 2020 (PNAS) — the core methodology**

The central idea: rather than running full N-body integrations for 10⁹ orbits (hours of compute), SPOCK runs a short 10⁴-orbit N-body integration to generate a set of ten dynamically motivated summary features — two involving the MEGNO chaos indicator, and the remaining eight derived from analytical two-planet mean-motion resonance (MMR) models — which are then passed to a gradient-boosted decision tree classifier that returns an estimated probability of stability over 10⁹ orbits. This works because for Neptune-mass-or-smaller planets, short-timescale instabilities are still dominantly driven by the overlap of two-body mean motion resonances. The model combines MEGNO with features capturing MMR dynamics, substantially improving on either indicator alone — MEGNO alone is a good indicator of instability but a poor one for confirming long-term stability, since chaos indicators can't see instabilities that develop past the short integration window. The net effect is a model roughly five orders of magnitude faster than direct N-body integration that generalizes well beyond its resonant training set to non-resonant and higher-multiplicity systems.

**3. Three-Body Stability tutorial (exoplanet-binary docs, by Billy Quarles)**

This page covers the physics you'd need to generate labeled training data for a *binary-star* analogue of SPOCK. Key points:

- Planets in binaries come in **S-Type** (orbiting one star), **P-Type** (circumbinary, orbiting both), and **T-Type** (Trojan) configurations.
- The classical approach (Holman & Wiegert 1999) defines a critical semimajor-axis ratio ρ_cr separating bound from scattered/ejected orbits, fit as a polynomial in mass ratio μ and binary eccentricity e_bin — but this breaks down at extreme mass ratios and for inclined/non-coplanar orbits.
- Quarles et al. (2020) extended this with wider μ coverage and planet inclination as an added dimension — pointing at a natural feature set: μ, e_bin, ρ = a_p/a_bin, inclination, and initial phase.
- The worked `rebound` examples (Kepler-16b for P-Type, γ Cephei Ab for S-Type) show how eccentricity/inclination vectors decompose into free and forced components, and how the Lidov-Kozai mechanism can flip a planet into a retrograde, unstable configuration over ~10⁴–10⁵ years — this is exactly the kind of longer-timescale behavior a short integration + ML classifier would need to predict without actually running it out.

Given your ETAMU move, it's worth noting Billy Quarles — the author of this tutorial and the source of the ρ_cr stability grids — is on the ETAMU Physics & Astronomy faculty doing exactly this kind of binary/circumbinary dynamics work, which could make for a very natural advisor or collaborator connection for this project.

## Suggested ML Frameworks

**1. scikit-learn + XGBoost (tabular / gradient-boosted trees) — start here.**
This is literally what the original FeatureClassifier uses, and for good reason: your inputs are a modest number of engineered dynamical features (MEGNO-like chaos indicators, MMR proximity, plus your binary-specific features like μ, e_bin, ρ, inclination), and your output is a stability probability — a textbook tabular classification problem. Gradient-boosted trees consistently outperform deep nets on this kind of small-to-medium tabular data with heterogeneous feature scales, train in seconds to minutes, and give you feature importances for free — valuable for a physics paper where you want to explain *why* a configuration is unstable, not just predict that it is. It's also the lowest-friction path if you want the community to trust and reuse your package (public, well-documented, minimal dependency footprint compared to a full deep learning stack).

**2. PyTorch (+ PyTorch Lightning) — for the deep regressor / uncertainty-aware extension.**
Once you have a working classifier, the more scientifically interesting (and more citable) contribution is often the regression/uncertainty side — predicting *when* instability sets in and how confident the model is, the way Cranmer et al.'s DeepRegressor does. PyTorch is the natural choice here because: (a) it's what SPOCK's own DeepRegressor is built on, so your code stays interoperable and comparable; (b) it plays well with `rebound`/`einops`-style feature engineering pipelines already used in this subfield; (c) it gives you access to techniques like deep ensembles or mixture-density outputs to get calibrated uncertainty on instability-time predictions — arguably the most useful thing to a binary-system observer trying to decide whether a candidate configuration is worth follow-up.

*A third option worth a mention, not a full recommendation*: if you want a lighter, more physics-native alternative to XGBoost for a first pass, `scikit-learn`'s `RandomForestClassifier`/`GradientBoostingClassifier` require zero extra dependencies (no separate xgboost install/OpenMP headaches on Mac) and are often within a percent or two of XGBoost's performance for datasets this size — a reasonable simplification for a public package aimed at broad accessibility.

## Next Steps

1. **Nail down the physical scope**: decide S-Type vs. P-Type first (they have very different feature sets and stability boundaries) — S-Type is probably the more tractable starting point given the γ Cephei example and closer analogy to existing single-star SPOCK features.
2. **Generate training data**: use `rebound` to run a large grid of short (~10³–10⁴ binary-orbit) integrations across (μ, e_bin, ρ, inclination, phase), labeling each with a long-integration (10⁵–10⁶+ orbit) ground-truth stability outcome — mirroring the Holman & Wiegert / Quarles grid approach but at higher resolution and with ML-ready feature extraction (MEGNO, forced eccentricity/inclination amplitudes, resonance-proximity metrics) rather than a simple ρ_cr cutoff.
3. **Reproduce SPOCK's feature set first**, then extend it: start by computing the same 10 MEGNO/MMR-based features SPOCK uses, see how well they transfer to the binary case, and only then add binary-specific ones (forced eccentricity, Kozai-relevant mutual inclination, μ, e_bin).
4. **Baseline with XGBoost**, get a working classifier and honest cross-validated AUC before touching deep learning.
5. **Validate against known systems** (Kepler-16b, γ Cephei Ab, and other confirmed circumbinary/S-Type planets) as sanity checks, not just synthetic grid points.
6. **Package for public use early**: set up the repo structure (pip-installable, `pytest` test suite, example Jupyter notebooks, Colab/Binder links) mirroring SPOCK's own layout from day one — it's much easier to build this in than retrofit later, and it's exactly what got SPOCK adopted community-wide.
7. Consider reaching out to Quarles or Tamayo directly once you have a working prototype — this is a natural, well-scoped extension of both of their published work and the kind of thing that gets a collaborative response.
```

Cautions:
- 

---
