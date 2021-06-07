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
import numpy as np


def fc(Rij, iRc, fRc):
    if Rij <= fRc and Rij > iRc:
        return 1
    else:
        return 0


def distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)


def G1(i, eta, Rs, iRc, fRc, structure):
    sum = 0
    neighbors = structure.get_neighbors(site=structure[i], r=fRc)
    ai = structure[i].species.elements[0].symbol
    for j in range(len(neighbors)):
        aj = neighbors[j].species.elements[0].symbol
        Rij = distance(structure[i].coords, neighbors[j].coords)
        sum += np.exp(-eta*(Rij-Rs)**2)*fc(Rij, iRc, fRc)
    return sum/len(structure)


def G2(i, eta, zeta, Rs, iRc, fRc, structure):
    sum = 0
    neighbors = structure.get_neighbors(site=structure[i], r=fRc)
    # ai = structure[i].species.elements[0].symbol
    for j in range(len(neighbors)):
        # aj = neighbors[j].species.elements[0].symbol
        Rij = distance(structure[i].coords, neighbors[j].coords)
        sum += np.exp(-eta*(Rij-Rs)**2)*fc(Rij, iRc, fRc)*np.exp(-zeta *
                                                                 np.abs(neighbors[j].species.elements[0].Z-structure[i].species.elements[0].Z))
    return sum/len(structure)


def descriptors(cif):
    parser = CifParser.from_string(cif)

    structure = parser.get_structures()
    structure = structure[0]

    # bonds_df = pd.read_csv('bonds.dat', sep='\s+')

    # For dataset_A.csv
    # G1_parameters = {"eta": [2, 4, 6], "Rs": [0, 2, 4], "Rc": [
    #     [0., 2], [2, 4], [4, 6]]}

    # For dataset_B.csv
    G1_parameters = {"eta": [1, 2, 3, 4, 5, 6], "Rs": [0, 1, 2, 3, 4], "Rc": [
        [0., 2], [2, 3], [3, 4], [4, 6]]}

    # For dataset_A.csv
    # G2_parameters = {"eta": [2, 4, 6], "zeta": [2, 4], "Rs": [0, 2, 4], "Rc": [
    #     [0., 2], [2, 4], [4, 6]]}

    # For dataset_B.csv
    G2_parameters = {"eta": [1, 2, 3, 4, 5, 6], "zeta": [1, 2, 3, 4], "Rs": [0, 1, 2, 3, 4], "Rc": [
        [0., 2], [2, 3], [3, 4], [4, 6]]}

    G1_descriptors = []
    G2_descriptors = []

    for eta in G1_parameters["eta"]:
        for Rs in G1_parameters["Rs"]:
            for Rc in G1_parameters["Rc"]:
                G = 0
                for i in range(len(structure)):
                    G += G1(i, eta, Rs, Rc[0], Rc[1], structure)
                G1_descriptors += [G]

    for eta in G2_parameters["eta"]:
        for zeta in G2_parameters["zeta"]:
            for Rs in G2_parameters["Rs"]:
                for Rc in G2_parameters["Rc"]:
                    G = 0
                    for i in range(len(structure)):
                        G += G2(i, eta, zeta, Rs, Rc[0], Rc[1], structure)
                    G2_descriptors += [G]

    descriptors_list = G1_descriptors + G2_descriptors
    # descriptors_list = G2_descriptors

    return descriptors_list
