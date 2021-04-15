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
import importlib
import json
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from ase.io import read
import pandas as pd


def descriptors(cif):
    f = open('a.cif', 'w')
    f.write(cif)
    f.flush()
    f.close

    a = read('a.cif')
    try:
        calc = GPAW(mode='lcao',
                    xc='PBE',
                    maxiter=1,
                    # nbands=100,
                    # parallel={'domain': 1, 'band': 1},
                    # occupations=FermiDirac(width=0.01),
                    convergence={'density': 1},
                    kpts=[1, 1, 1],
                    txt='gs.txt')

        # a.set_calculator(calc)
        calc.calculate(a)
        # a.get_potential_energy()
        ev = pd.DataFrame(calc.get_eigenvalues())
        ef = calc.get_fermi_level()
        ev_below = ev.loc[ev.values <= ef]
        ev_above = ev.loc[ev.values > ef]
        descriptors_below = []
        descriptors_above = []

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
        return descriptors_below + descriptors_above
    except:
        print('Problem in material')
        
