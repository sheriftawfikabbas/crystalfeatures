# !pip3 install pymatgen
# !pip3 install xgboost
# !pip3 install sklearn pandas
from pymatgen.io.cif import CifParser
from urllib.request import urlopen
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.ext.matproj import MPRestError
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np


def distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)


def descriptors(cif):
    parser = CifParser.from_string(cif)

    structure = parser.get_structures()
    structure = structure[0]

    # bonds_df = pd.read_csv('bonds.dat', sep='\s+')

    fragments = {}
    dipoles = {}
    asymmetry = {}
    for i in range(len(structure)):
        neighbors = structure.get_neighbors(site=structure[i], r=3)
        ai = structure[i].species.elements[0].symbol
        f = []
        for j in range(len(neighbors)):
            aj = neighbors[j].species.elements[0].symbol
            # bond = bonds_df.query(
            #     '(atom1=="'+ai+'" and atom2=="'+aj+'") or (atom1=="'+aj+'" and atom2=="'+ai+'") ')
            # if bond.shape[0] == 0:
            #     bond = 0
            # else:
            #     bond = bond.bond.values[0]
            b1,b2=0,0
            if neighbors[j].species.elements[0].atomic_radius != None:
                b1=neighbors[j].species.elements[0].atomic_radius
            else:
                b1=neighbors[j].species.elements[0].van_der_waals_radius
            if structure[i].species.elements[0].atomic_radius != None:
                b2=structure[i].species.elements[0].atomic_radius
            else:
                b2=structure[i].species.elements[0].van_der_waals_radius
            bond = b1 + b2
            if abs(distance(structure[i].coords, neighbors[j].coords)-bond) < 5e-1:
                f += [neighbors[j]]
        fragments[i] = f
        ri = np.array(structure[i].coords)
        p = np.array([0.0, 0.0, 0.0])
        d = 0
        for aj in f:
            rj = np.array(aj.coords)
            pj = (rj-ri)*aj.species.elements[0].Z
            p += pj
            d += distance(ri, rj)*aj.species.elements[0].Z
        asymmetry[i] = np.dot(p, p)
        dipoles[i] = d

        if asymmetry[i] > 0:
            # asymmetry[i] = np.log(asymmetry[i])
            asymmetry[i] = asymmetry[i]
        if dipoles[i] > 0:
            # dipoles[i] = np.log(dipoles[i])
            dipoles[i] = dipoles[i]

    elements_asymmetry = {}

    for i in range(1, 95):
        elements_asymmetry[i] = 0

    for i in range(len(structure)):
        elements_asymmetry[structure[i].species.elements[0].Z] += asymmetry[i]

    elements_asymmetry_list = []
    for i in range(1, 95):
        elements_asymmetry_list += [elements_asymmetry[i]]

    elements_dipoles = {}

    for i in range(1, 95):
        elements_dipoles[i] = 0

    for i in range(len(structure)):
        elements_dipoles[structure[i].species.elements[0].Z] += dipoles[i]

    elements_dipoles_list = []
    for i in range(1, 95):
        elements_dipoles_list += [elements_dipoles[i]]

    descriptors_list = \
        elements_asymmetry_list +\
        elements_dipoles_list

    return descriptors_list
