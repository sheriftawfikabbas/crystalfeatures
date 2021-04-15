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

def descriptors(cif):

    atomic_numbers = []

    distance_matrix = []
    van_der_waals_radius = []
    electrical_resistivity = []
    velocity_of_sound = []
    reflectivity = []
    poissons_ratio = []
    molar_volume = []
    thermal_conductivity = []
    melting_point = []
    critical_temperature = []
    superconduction_temperature = []
    liquid_range = []
    bulk_modulus = []
    youngs_modulus = []
    brinell_hardness = []
    rigidity_modulus = []
    # mineral_hardness = []
    vickers_hardness = []
    density_of_solid = []
    coefficient_of_linear_thermal_expansion = []
    average_ionic_radius = []
    average_cationic_radius = []
    average_anionic_radius = []

    parser = CifParser.from_string(cif)

    structure = parser.get_structures()
    structure = structure[0]

    numElements = len(structure.atomic_numbers)

    num_metals = 0
    for e in structure.species:
        if e.Z in range(3, 4+1) or e.Z in range(11, 12+1) or e.Z in range(19, 30+1) or e.Z in range(37, 48+1) or e.Z in range(55, 80 + 1) or e.Z in range(87, 112+1):
            num_metals += 1
    metals_fraction = num_metals/numElements

    spg = structure.get_space_group_info()

    spacegroup_numbers = {}
    for i in range(1, 231):
        spacegroup_numbers[i] = 0

    spacegroup_numbers[spg[1]] = 1

    spacegroup_numbers_list = []
    for i in range(1, 231):
        spacegroup_numbers_list += [spacegroup_numbers[i]]

    atomic_numbers = [np.mean(structure.atomic_numbers), np.max(structure.atomic_numbers), np.min(
        structure.atomic_numbers), np.std(structure.atomic_numbers)]

    # Lattice parameters:
    a_parameters = structure.lattice.abc[0]
    b_parameters = structure.lattice.abc[1]
    c_parameters = structure.lattice.abc[2]
    alpha_parameters = structure.lattice.angles[0]
    beta_parameters = structure.lattice.angles[1]
    gamma_parameters = structure.lattice.angles[2]

    distance_matrix += [np.mean(structure.distance_matrix), np.max(structure.distance_matrix),
                        np.min(structure.distance_matrix), np.std(structure.distance_matrix)]

    e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23 = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for e in structure.species:
        e1 += [e.van_der_waals_radius]
        e2 += [e.electrical_resistivity]
        e3 += [e.velocity_of_sound]
        e4 += [e.reflectivity]
        e6 += [e.poissons_ratio]
        e7 += [e.molar_volume]
        e8 += [e.thermal_conductivity]
        e9 += [e.melting_point]
        e10 += [e.critical_temperature]
        e11 += [e.superconduction_temperature]
        e12 += [e.liquid_range]
        e13 += [e.bulk_modulus]
        e14 += [e.youngs_modulus]
        e15 += [e.brinell_hardness]
        e16 += [e.rigidity_modulus]
        # e17 +=[e.mineral_hardness ]
        e18 += [e.vickers_hardness]
        e19 += [e.density_of_solid]
        e20 += [e.coefficient_of_linear_thermal_expansion]
        e21 += [e.average_ionic_radius]
        e22 += [e.average_cationic_radius]
        e23 += [e.average_anionic_radius]

    e1 = [0 if v is None else v for v in e1]
    e2 = [0 if v is None else v for v in e2]
    e3 = [0 if v is None else v for v in e3]
    e4 = [0 if v is None else v for v in e4]
    # e5=[0 if v is None else v for v in e5]
    e6 = [0 if v is None else v for v in e6]
    e7 = [0 if v is None else v for v in e7]
    e8 = [0 if v is None else v for v in e8]
    e9 = [0 if v is None else v for v in e9]
    e10 = [0 if v is None else v for v in e10]
    e11 = [0 if v is None else v for v in e11]
    e12 = [0 if v is None else v for v in e12]
    e13 = [0 if v is None else v for v in e13]
    e14 = [0 if v is None else v for v in e14]
    e15 = [0 if v is None else v for v in e15]
    e16 = [0 if v is None else v for v in e16]
    # e17=[0 if v is None else v for v in e17]
    e18 = [0 if v is None else v for v in e18]
    e19 = [0 if v is None else v for v in e19]
    e20 = [0 if v is None else v for v in e20]
    e21 = [0 if v is None else v for v in e21]
    e22 = [0 if v is None else v for v in e22]
    e23 = [0 if v is None else v for v in e23]

    van_der_waals_radius = [np.mean(e1), np.max(e1), np.min(e1), np.std(e1)]
    electrical_resistivity = [np.mean(e2), np.max(e2), np.min(e2), np.std(e2)]
    velocity_of_sound = [np.mean(e3), np.max(e3), np.min(e3), np.std(e3)]
    reflectivity = [np.mean(e4), np.max(e4), np.min(e4), np.std(e4)]
    poissons_ratio = [np.mean(e6), np.max(e6), np.min(e6), np.std(e6)]
    molar_volume = [np.mean(e7), np.max(e7), np.min(e7), np.std(e7)]
    thermal_conductivity = [np.mean(e8), np.max(e8), np.min(e8), np.std(e8)]
    melting_point = [np.mean(e9), np.max(e9), np.min(e9), np.std(e9)]
    critical_temperature = [np.mean(e10), np.max(
        e10), np.min(e10), np.std(e10)]
    superconduction_temperature = [
        np.mean(e11), np.max(e11), np.min(e11), np.std(e11)]
    liquid_range = [np.mean(e12), np.max(e12), np.min(e12), np.std(e12)]
    bulk_modulus = [np.mean(e13), np.max(e13), np.min(e13), np.std(e13)]
    youngs_modulus = [np.mean(e14), np.max(e14), np.min(e14), np.std(e14)]
    brinell_hardness = [np.mean(e15), np.max(e15), np.min(e15), np.std(e15)]
    rigidity_modulus = [np.mean(e16), np.max(e16), np.min(e16), np.std(e16)]
    vickers_hardness = [np.mean(e18), np.max(e18), np.min(e18), np.std(e18)]
    density_of_solid = [np.mean(e19), np.max(e19), np.min(e19), np.std(e19)]
    coefficient_of_linear_thermal_expansion = [
        np.mean(e20), np.max(e20), np.min(e20), np.std(e20)]
    average_ionic_radius = [np.mean(e21), np.max(
        e21), np.min(e21), np.std(e21)]
    average_cationic_radius = [
        np.mean(e22), np.max(e22), np.min(e22), np.std(e22)]
    average_anionic_radius = [
        np.mean(e23), np.max(e23), np.min(e23), np.std(e23)]

    V = a_parameters*b_parameters*c_parameters
    Density = V / numElements

    descriptors_list = atomic_numbers +\
        [Density] +\
        [alpha_parameters] +\
        [beta_parameters] +\
        [gamma_parameters] +\
        [metals_fraction] +\
        distance_matrix +\
        van_der_waals_radius +\
        electrical_resistivity +\
        velocity_of_sound +\
        reflectivity +\
        poissons_ratio +\
        molar_volume +\
        thermal_conductivity +\
        melting_point +\
        critical_temperature +\
        superconduction_temperature +\
        liquid_range +\
        bulk_modulus +\
        youngs_modulus +\
        brinell_hardness +\
        rigidity_modulus +\
        vickers_hardness +\
        density_of_solid +\
        coefficient_of_linear_thermal_expansion +\
        average_ionic_radius +\
        average_cationic_radius +\
        average_anionic_radius 
        # spacegroup_numbers_list

    # dataset_df = pd.DataFrame({
    #     "mean_atomic_numbers": atomic_numbers[0],
    #     "max_atomic_numbers": atomic_numbers[1],
    #     "min_atomic_numbers": atomic_numbers[2],
    #     "std_atomic_numbers": atomic_numbers[3],
    #     "a_parameters": a_parameters,
    #     "b_parameters": b_parameters,
    #     "c_parameters": c_parameters,
    #     "V": V,
    #     "alpha_parameters": alpha_parameters,
    #     "beta_parameters": beta_parameters,
    #     "gamma_parameters": gamma_parameters,
    #     "mean_distance_matrix": distance_matrix[0],
    #     "max_distance_matrix": distance_matrix[1],
    #     "min_distance_matrix": distance_matrix[2],
    #     "std_distance_matrix": distance_matrix[3],
    #     "mean_van_der_waals_radius": van_der_waals_radius[0],
    #     "max_van_der_waals_radius": van_der_waals_radius[1],
    #     "min_van_der_waals_radius": van_der_waals_radius[2],
    #     "std_van_der_waals_radius": van_der_waals_radius[3],
    #     "mean_electrical_resistivity": electrical_resistivity[0],
    #     "max_electrical_resistivity": electrical_resistivity[1],
    #     "min_electrical_resistivity": electrical_resistivity[2],
    #     "std_electrical_resistivity": electrical_resistivity[3],
    #     "mean_velocity_of_sound": velocity_of_sound[0],
    #     "max_velocity_of_sound": velocity_of_sound[1],
    #     "min_velocity_of_sound": velocity_of_sound[2],
    #     "std_velocity_of_sound": velocity_of_sound[3],
    #     "mean_reflectivity": reflectivity[0],
    #     "max_reflectivity": reflectivity[1],
    #     "min_reflectivity": reflectivity[2],
    #     "std_reflectivity": reflectivity[3],
    #     "mean_poissons_ratio": poissons_ratio[0],
    #     "max_poissons_ratio": poissons_ratio[1],
    #     "min_poissons_ratio": poissons_ratio[2],
    #     "std_poissons_ratio": poissons_ratio[3],
    #     "mean_molar_volume": molar_volume[0],
    #     "max_molar_volume": molar_volume[1],
    #     "min_molar_volume": molar_volume[2],
    #     "std_molar_volume": molar_volume[3],
    #     "mean_thermal_conductivity": thermal_conductivity[0],
    #     "max_thermal_conductivity": thermal_conductivity[1],
    #     "min_thermal_conductivity": thermal_conductivity[2],
    #     "std_thermal_conductivity": thermal_conductivity[3],
    #     "mean_melting_point": melting_point[0],
    #     "max_melting_point": melting_point[1],
    #     "min_melting_point": melting_point[2],
    #     "std_melting_point": melting_point[3],
    #     "mean_critical_temperature": critical_temperature[0],
    #     "max_critical_temperature": critical_temperature[1],
    #     "min_critical_temperature": critical_temperature[2],
    #     "std_critical_temperature": critical_temperature[3],
    #     "mean_superconduction_temperature": superconduction_temperature[0],
    #     "max_superconduction_temperature": superconduction_temperature[1],
    #     "min_superconduction_temperature": superconduction_temperature[2],
    #     "std_superconduction_temperature": superconduction_temperature[3],
    #     "mean_liquid_range": liquid_range[0],
    #     "max_liquid_range": liquid_range[1],
    #     "min_liquid_range": liquid_range[2],
    #     "std_liquid_range": liquid_range[3],
    #     "mean_bulk_modulus": bulk_modulus[0],
    #     "max_bulk_modulus": bulk_modulus[1],
    #     "min_bulk_modulus": bulk_modulus[2],
    #     "std_bulk_modulus": bulk_modulus[3],
    #     "mean_youngs_modulus": youngs_modulus[0],
    #     "max_youngs_modulus": youngs_modulus[1],
    #     "min_youngs_modulus": youngs_modulus[2],
    #     "std_youngs_modulus": youngs_modulus[3],
    #     "mean_brinell_hardness": brinell_hardness[0],
    #     "max_brinell_hardness": brinell_hardness[1],
    #     "min_brinell_hardness": brinell_hardness[2],
    #     "std_brinell_hardness": brinell_hardness[3],
    #     "mean_rigidity_modulus": rigidity_modulus[0],
    #     "max_rigidity_modulus": rigidity_modulus[1],
    #     "min_rigidity_modulus": rigidity_modulus[2],
    #     "std_rigidity_modulus": rigidity_modulus[3],
    #     "mean_vickers_hardness": vickers_hardness[0],
    #     "max_vickers_hardness": vickers_hardness[1],
    #     "min_vickers_hardness": vickers_hardness[2],
    #     "std_vickers_hardness": vickers_hardness[3],
    #     "mean_density_of_solid": density_of_solid[0],
    #     "mean_coefficient_of_linear_thermal_expansion": coefficient_of_linear_thermal_expansion[0],
    #     "max_coefficient_of_linear_thermal_expansion": coefficient_of_linear_thermal_expansion[1],
    #     "min_coefficient_of_linear_thermal_expansion": coefficient_of_linear_thermal_expansion[2],
    #     "std_coefficient_of_linear_thermal_expansion": coefficient_of_linear_thermal_expansion[3],
    #     "mean_average_ionic_radius": average_ionic_radius[0],
    #     "max_average_ionic_radius": average_ionic_radius[1],
    #     "min_average_ionic_radius": average_ionic_radius[2],
    #     "std_average_ionic_radius": average_ionic_radius[3],
    #     "mean_average_cationic_radius": average_cationic_radius[0],
    #     "max_average_cationic_radius": average_cationic_radius[1],
    #     "min_average_cationic_radius": average_cationic_radius[2],
    #     "std_average_cationic_radius": average_cationic_radius[3],
    #     "mean_average_anionic_radius": average_anionic_radius[0],
    #     "max_average_anionic_radius": average_anionic_radius[1],
    #     "min_average_anionic_radius": average_anionic_radius[2],
    #     "std_average_anionic_radius": average_anionic_radius[3]
    # })
    return descriptors_list
