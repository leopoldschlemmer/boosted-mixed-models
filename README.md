# Boosted Mixed Models

This repository contains the code and paper materials for experiments on conjugate non-Gaussian mixed-effects models, in particular Poisson--Gamma and Gamma--Gamma models implemented within the GPBoost framework.

## Repository Structure

- `Poisson_Gamma/`: code and experiments for the Poisson--Gamma models
- `Gamma_Gamma/`: code and experiments for the Gamma--Gamma models
- `...`: paper source and related materials

## Data

Some raw datasets are not included in the repository due to size and/or licensing constraints. The paper appendix documents the original data sources and preprocessing steps.

## Reproducibility

The experiments were run in Python using GPBoost together with standard scientific Python packages. Parts of the GPBoost source code were modified locally for the custom likelihood implementations.

## Notes

This repository is primarily intended to accompany the paper and document the experimental workflow.
