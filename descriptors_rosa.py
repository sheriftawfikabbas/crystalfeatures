from ase import Atoms
import os
import string
import random
from pymatgen.io.cif import CifParser
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.build import bulk, niggli_reduce
from gpaw import GPAW, PW, FermiDirac
from ase.io import read
import pandas as pd

import glob

folder = '/g/data/mo5/sat562/'
folder = '/mnt/c/MyCodes/MaterialScience/GPAW/'
g = glob.glob(
    folder + 'gpaw-setups-0.9.20000/*.PBE.gz')

available_atoms = []
for i in g:
    available_atoms += [i.replace(
        folder + 'gpaw-setups-0.9.20000/', '').split('.')[0]]


def fix_psuedo(a):
    for i in range(len(a)):
        if not a[i].symbol in available_atoms:
            print(a[i].symbol, 'is not available in GPAW, replacing it with Y')
            a[i].symbol = 'Y'


def get_descriptors_for_structure(a, descriptor_size=100):
    half_descriptor_size=int(descriptor_size/2)
    calc = GPAW(mode='lcao',
                xc='PBE',
                maxiter=1,
                # nbands=100,
                # parallel={'domain': 1, 'band': 1},
                # occupations=FermiDirac(width=0.01),
                convergence={'density': 1},
                kpts=[1, 1, 1],
                txt='gpaw_lcao.txt')

    # a.set_calculator(calc)
    calc.calculate(a)
    H = calc.hamiltonian
    descriptors_below = []
    descriptors_above = []

    num_atoms = a.get_global_number_of_atoms()

    ev = pd.DataFrame(calc.get_eigenvalues())
    ef = calc.get_fermi_level()
    ev_below = ev.loc[ev.values <= ef]
    ev_above = ev.loc[ev.values > ef]

    for e in ev_below.values.tolist()[::-1]:
        descriptors_below += e
    if len(ev_below) > half_descriptor_size:
        descriptors_below = descriptors_below[0:half_descriptor_size]
    elif len(ev_below) < half_descriptor_size:
        for e in range(half_descriptor_size-len(ev_below)):
            descriptors_below += [0]
    descriptors_below = descriptors_below[::-1]

    for e in ev_above.values.tolist():
        descriptors_above += e
    if len(ev_above) > half_descriptor_size:
        descriptors_above = descriptors_above[0:half_descriptor_size]
    elif len(ev_above) < half_descriptor_size:
        for e in range(half_descriptor_size-len(ev_above)):
            descriptors_above += [0]

    return descriptors_below + descriptors_above + \
        [ef,
         H.e_band/num_atoms,
         H.e_coulomb/num_atoms,
         H.e_entropy/num_atoms,
         H.e_external/num_atoms,
         H.e_kinetic/num_atoms,
         H.e_kinetic0/num_atoms,
         H.e_xc/num_atoms,
         H.e_total_free/num_atoms]


def descriptors(cif, descriptor_size=100, calculation_type='bulk'):
    if type(cif) is string:
        parser = CifParser.from_string(cif)
        structure = parser.get_structures()
        structure = structure[0]

        a = Atoms(pbc=True, cell=structure.lattice.matrix,
                positions=structure.cart_coords, numbers=structure.atomic_numbers)
    else:
        a = cif
    d_pristine = []
    try:
        a_copy = a.copy()
        if calculation_type == 'bulk':
            niggli_reduce(a)
        fix_psuedo(a)
        d_pristine = get_descriptors_for_structure(a, descriptor_size)
        
    except Exception as e:
        print('Problem in GPAW')
    if len(d_pristine) > 0:
        return d_pristine
    else:
        return []
