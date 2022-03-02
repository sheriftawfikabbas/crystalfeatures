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

g = glob.glob(
    '/mnt/c/MyCodes/MaterialScience/GPAW/gpaw-setups-0.9.20000/*.PBE.gz')

available_atoms = []
for i in g:
    available_atoms += [i.replace(
        '/mnt/c/MyCodes/MaterialScience/GPAW/gpaw-setups-0.9.20000/', '').split('.')[0]]


def fix_psuedo(a):
    for i in range(len(a)):
        if not a[i].symbol in available_atoms:
            print(a[i].symbol, 'is not available in GPAW, replacing it with Y')
            a[i].symbol = 'Y'


def get_descriptors_for_structure(a):
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
    if len(ev_below) > 50:
        descriptors_below = descriptors_below[0:50]
    elif len(ev_below) < 50:
        for e in range(50-len(ev_below)):
            descriptors_below += [0]
    descriptors_below = descriptors_below[::-1]

    for e in ev_above.values.tolist():
        descriptors_above += e
    if len(ev_above) > 50:
        descriptors_above = descriptors_above[0:50]
    elif len(ev_above) < 50:
        for e in range(50-len(ev_above)):
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


def descriptors(cif):
    import random
    import string
    import os
    file_name = ''.join(random.choices(string.ascii_uppercase +
                                       string.digits, k=10))
    f = open(file_name+'.cif', 'w')
    f.write(cif)
    f.flush()
    f.close

    d_pristine = []
    d_changed = []

    try:

        a = read(file_name+'.cif')
        niggli_reduce(a)
        fix_psuedo(a)
        d_pristine = get_descriptors_for_structure(a)
        # a.get_potential_energy()

        # Replace heaviest atom with atomic_number-1, lightest with atomic_number+1
        a = read(file_name+'.cif')
        niggli_reduce(a)
        fix_psuedo(a)
        max_atomic_number = max(a.numbers)
        min_atomic_number = min(a.numbers)

        if max_atomic_number != max_atomic_number:
            for i in range(len(a)):
                if a[i].number == max_atomic_number:
                    a[i].number -= 1
                elif a[i].number == min_atomic_number:
                    a[i].number += 1

        d_changed = get_descriptors_for_structure(a)
        os.remove(file_name+'.cif')
    except Exception as e:
        print('Problem in GPAW')
        os.remove(file_name+'.cif')
    if len(d_changed) > 0 and len(d_pristine) > 0:
        return d_pristine + d_changed
    else:
        return []
