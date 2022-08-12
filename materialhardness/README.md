# MaterialHardNess

## Setup

Download and extract julia `https://julialang.org/downloads/` to your home folder ($HOME)
From a bash shell, run the following commands to install the dependencies:

```
$ export JULIA_HOME=$HOME/julia-1.7.3
$ export PATH=$JULIA_HOME/bin:$PATH
$ cd MaterialHardNess
(MaterialHardNess) $ julia
julia> ]
(v1.7) pkg> activate .
(MaterialHardNess) pkg> instantiate
(MaterialHardNess) pkg> <backspace>
julia> using Pkg, Conda, PyCall
julia> Conda.add("scikit-learn")
julia> Conda.add("xgboost")
julia> ENV["PYTHON"] = "$(ENV["HOME"])/.julia/conda/3/python"
julia> Pkg.build("PyCall")
julia> exit()
$ ~/.julia/conda/3/bin/conda install libgcc
$ export LD_LIBRARY_PATH=$HOME/.julia/conda/3/lib:$LD_LIBRARY_PATH
```

Put the following into your ~/.bashrc so that you can run julia from bash shell

```
export JULIA_HOME=$HOME/julia-1.7.3
export PATH=$JULIA_HOME/bin:$HOME/.julia/conda/3/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.julia/conda/3/lib:$LD_LIBRARY_PATH
```

## Dataset

Data path: `~/datasets/materials`:
    ```
     664K  dataset_elasticity.csv
      17M  dataset_shsdft_elasticity.csv
     193M  full_dataset.csv
     205M  full_dataset_updated.csv
    ```

Input: full_dataset_updated.csv. There are 2235 features altogether.  The features available are:

- From column B to DF: pristine SDFT, 0% strain
- DG to HK: defective SDFT, 0% strain
- HL to AXI: pristine SDFT with strains from -5% to 5% in steps of 1% (without the 0% point)
- AXJ to BJX: atomic descriptors
- BJY to the end: geometry descriptors

Target: dataset_elasticity.csv. The sheet with the "target" values that we wish to predict. The two most important ones are G_VRH and K_VRH (shear and bulk modulus, respectively). You can merge the two CSV files via "material_id".

- bulk modulus (K_VRH column), continuous
- shear modulus (G_VRH column), continuous


## Experiment scripts

```
$ julia --project=. xgboost-params-search.jl
$ julia --project=. xgboost.jl
```

File xgboost.jl train an xgboost regressor on 4 folds and test on 1 fold in a 5-fold split:

    ```julia
    get_data = data_update  # function to get data (X, y)
    d1 = xgboost_each_fold(; get_data)  # run 5 fold CV and report R2 score for each fold
    d2 = xgboost_each_feature(; get_data)  # run on a 5-fold split and report R2 score for each feature mention above. This is for reporting feature importance
    ```
