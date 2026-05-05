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

The compiled GPBoost shared libraries (`lib_gpboost.so`) are generated build artifacts and are intentionally not tracked in Git. A fresh clone therefore requires compiling the modified GPBoost sources before running the experiments.

### Build the modified GPBoost libraries

Each experiment folder contains its own modified GPBoost source tree:

- `Poisson_Gamma/GPBoost_full backup/`
- `Gamma_Gamma/GPBoost_full backup/`

On macOS with Homebrew `boost` and `libomp`, the build command used for the experiments was:

```sh
cd "Gamma_Gamma/GPBoost_full backup"
mkdir -p build
cd build
cmake .. \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DUSE_GPU=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_MPI=OFF \
  -DCMAKE_C_FLAGS="-w -I/opt/homebrew/opt/boost/include" \
  -DCMAKE_CXX_FLAGS="-ferror-limit=1 -w -I/opt/homebrew/opt/boost/include" \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
make -j"$(sysctl -n hw.ncpu)"
```

Run the same build steps inside `Poisson_Gamma/GPBoost_full backup/` to compile the Poisson--Gamma version. On Linux, the same workflow applies, but the OpenMP and Boost paths may need to be adjusted or omitted depending on the system installation.

After compiling, return to the repository root and install the corresponding local Python package into the virtual environment used for each experiment:

```sh
cd Gamma_Gamma
python -m pip install -e "GPBoost_full backup/python-package"
python experiments_gammagamma.py
```

```sh
cd Poisson_Gamma
python -m pip install -e "GPBoost_full backup/python-package"
python experiments.py
```

The result tables used in the paper are written to `Gamma_Gamma/results/real_data_gg_summary.csv` and `Poisson_Gamma/results/real_data_experiment_pg_summary.csv`.

## Notes

This repository is primarily intended to accompany the paper and document the experimental workflow.
