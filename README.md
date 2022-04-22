# CrystalFeatures

CrystalFeatures is a set of python codes that extract features of crystal systems, using only the CIF file as input, for machine learning applications. There are three classes of features in CrystalFeatures: basic atomic and crystal descriptors (BACD), which include statistical properties of the atoms in the materials as well as the crystal symmetry; geometry features, which are based on bonding properties; and robust one-shot ab initio (ROSA) descriptors, where the eigen values of the material are obtained using a single-point single step calculation is performed using GPAW.

# Data for ROSA paper (https://doi.org/10.48550/arXiv.2203.03392)

In order to reproduce the results in the manuscript, the trained XGBOOST models are provided in the folder `/data/models/`, and the test set data are provided in the folder `/data/test/` for each of the properties listed in the paper.