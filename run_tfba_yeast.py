import time
import cobra
import numpy as np
import pandas as pd
from tqdm import tqdm

from ThermoInfer.utils.constants import *
from ThermoInfer.utils.func import tGEM


# -----------------------------
# User-defined file paths
# -----------------------------

gem_path = "./yeast-GEM.xml"
dgr_path = "./Yeast9_standard_dGr_dGbyG.csv"

tfba_output_path = "./Yeast9_Directionality_TFBA.csv"


# -----------------------------
# User-defined run parameters
# -----------------------------

batch_size = 40
thread = 10
biomass_fraction = 0.1


# -----------------------------
# Compartment-specific conditions
# -----------------------------

compartment_conditions = {
    "c":  {"pH": 7.2,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "e":  {"pH": 7.0,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "g":  {"pH": 6.35, "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "m":  {"pH": 7.5,  "e_potential": -0.155, "T": 298.15, "I": 0.25, "pMg": 14.0},
    "n":  {"pH": 7.2,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "v":  {"pH": 6.2,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "ce": {"pH": 7.0,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "p":  {"pH": 7.4,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "er": {"pH": 7.2,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "lp": {"pH": 7.0,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "erm": {"pH": 7.2, "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "vm": {"pH": 6.2,  "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "gm": {"pH": 6.35, "e_potential": 0.0,    "T": 298.15, "I": 0.25, "pMg": 14.0},
    "mm": {"pH": 7.5,  "e_potential": -0.155, "T": 298.15, "I": 0.25, "pMg": 14.0},
}


# -----------------------------
# Load GEM
# -----------------------------

print("Loading Yeast-GEM...")
gem = cobra.io.read_sbml_model(gem_path)

print(f"Number of reactions: {len(gem.reactions)}")
print(f"Number of metabolites: {len(gem.metabolites)}")


# -----------------------------
# Reset reaction bounds
# -----------------------------

print("Resetting non-boundary reaction bounds...")

for rxn in gem.reactions:
    if not rxn.boundary:
        rxn.lower_bound = -1000
        rxn.upper_bound = 1000


# -----------------------------
# Assign metabolite concentration ranges
# -----------------------------

print("Assigning metabolite concentration ranges...")

for met in tqdm(gem.metabolites):
    if met.formula == "H2O":
        met.lz = H2O_lz
        met.uz = H2O_uz

    elif met.formula == "H":
        met.lz = -compartment_conditions[met.compartment]["pH"] - 1.0
        met.uz = -compartment_conditions[met.compartment]["pH"] + 1.0

    else:
        met.lz = default_lz
        met.uz = default_uz


# -----------------------------
# Load dGbyG reaction-level predictions
# -----------------------------

print("Loading dGbyG reaction-level predictions...")

Rxn_df = pd.read_csv(dgr_path, index_col=0)
dGr = Rxn_df[["dGr_prime", "SD of dGr_prime"]].to_numpy()

single_compartment_rxn = np.array([
    len(rxn.compartments) == 1
    for rxn in gem.reactions
])

dGr[~single_compartment_rxn, :] = np.nan

print(f"Reactions with usable dGr_prime values: {np.sum(~np.isnan(dGr[:, 0]))}")


# -----------------------------
# Build ThermoInfer model
# -----------------------------

print("Building ThermoInfer model...")

max_biomass = gem.slim_optimize()
biomass_synthesis = biomass_fraction * max_biomass

print(f"Maximum biomass flux: {max_biomass}")
print(f"Minimal biomass requirement: {biomass_synthesis}")

tgem = tGEM(
    GEM=gem,
    dGr=dGr,
    concentration_ub=None,
    biomass_synthesis=biomass_synthesis
)


# -----------------------------
# Run TFBA directionality inference
# -----------------------------

print("Running TFBA directionality inference...")

t0 = time.time()

tgem.TFBA_res_file_path = tfba_output_path
tgem.concurrent_infer_v_and_dGr(
    v_si=0,
    v_ei=None,
    batch_size=batch_size,
    thread=thread
)

print(f"TFBA finished in {(time.time() - t0) / 60:.2f} min")

