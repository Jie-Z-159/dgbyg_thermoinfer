import time
import json
import argparse
import os
import contextlib
import cobra
import numpy as np
import pandas as pd
from tqdm import tqdm

from ThermoInfer.utils.constants import *
from ThermoInfer.utils.func import tGEM


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def get_default_compartment_conditions(gem):
    return {
        compartment: {"pH": default_pH, "e_potential": 0.0, "T": default_T, "I": default_I, "pMg": default_pMg}
        for compartment in gem.compartments
    }


def load_compartment_conditions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run TFBA directionality inference on a genome-scale metabolic model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Supported GEM file formats:
  .xml / .sbml    SBML format, read with cobra.io.read_sbml_model()
  .mat            MATLAB format, read with cobra.io.load_matlab_model()

Example usage:
  python run_tfba.py model.xml dGr_predictions.csv
  python run_tfba.py model.mat dGr_predictions.csv --compartments compartments.json
  python run_tfba.py model.xml dGr_predictions.csv --output results.csv --threads 16
        '''
    )

    parser.add_argument('gem_path', type=str,
                        help='Path to the GEM file (.xml, .sbml, or .mat)')
    parser.add_argument('dgr_path', type=str,
                        help='Path to the dGbyG reaction-level predictions CSV file')
    parser.add_argument('--compartments', type=str, default=None,
                        help='Path to compartment conditions JSON file. If not provided, all compartments use default values (pH=7.0, e_potential=0.0, T=298.15, I=0.25, pMg=14.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output TFBA directionality CSV file (default: <gem_basename>_Directionality_TFBA.csv)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size for parallel processing (default: 20)')
    parser.add_argument('--threads', type=int, default=5,
                        help='Number of threads for parallel processing (default: 5)')
    parser.add_argument('--biomass-fraction', type=float, default=0.1,
                        help='Fraction of maximum biomass flux to use as minimum requirement (default: 0.1)')
    parser.add_argument('--v-si', type=int, default=0,
                        help='Start reaction index (0-based) for TFBA inference. Use to resume an interrupted run or process a subset of reactions (default: 0)')
    parser.add_argument('--v-ei', type=int, default=None,
                        help='End reaction index (0-based, inclusive) for TFBA inference. Use to process a subset of reactions. If not provided, runs to the last reaction (default: None)')

    parser.add_argument('--run-fba', action='store_true', default=False,
                        help='Also run FBA directionality inference and save to <gem_basename>_Directionality_FBA.csv')

    return parser.parse_args()


args = parse_arguments()

gem_path = args.gem_path
dgr_path = args.dgr_path

if args.output is None:
    gem_basename = os.path.splitext(os.path.basename(gem_path))[0]
    tfba_output_path = f"./{gem_basename}_Directionality_TFBA.csv"
    fba_output_path = f"./{gem_basename}_Directionality_FBA.csv"
else:
    tfba_output_path = args.output
    fba_output_path = os.path.splitext(args.output)[0].replace('_TFBA', '') + '_FBA.csv'

batch_size = args.batch_size
thread = args.threads
biomass_fraction = args.biomass_fraction
v_si = args.v_si
v_ei = args.v_ei

compartment_conditions_source = args.compartments


# -----------------------------
# Load GEM
# -----------------------------

print(f"Loading GEM from {gem_path}...")
if gem_path.endswith(".xml") or gem_path.endswith(".sbml"):
    gem = cobra.io.read_sbml_model(gem_path)
elif gem_path.endswith(".mat"):
    gem = cobra.io.load_matlab_model(gem_path)
else:
    raise ValueError(f"Unsupported GEM file format: '{gem_path}'. Please provide a .xml, .sbml, or .mat file.")

print(f"Number of reactions: {len(gem.reactions)}")
print(f"Number of metabolites: {len(gem.metabolites)}")

if compartment_conditions_source is None:
    print("Using default compartment conditions...")
    compartment_conditions = get_default_compartment_conditions(gem)
else:
    print(f"Loading compartment conditions from {compartment_conditions_source}...")
    compartment_conditions = load_compartment_conditions(compartment_conditions_source)


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

with suppress_stdout_stderr():
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
    v_si=v_si,
    v_ei=v_ei,
    batch_size=batch_size,
    thread=thread
)

print(f"TFBA finished in {(time.time() - t0) / 60:.2f} min")


# -----------------------------
# Run FBA directionality inference (optional)
# -----------------------------

if args.run_fba:
    print("Running FBA directionality inference...")
    t0 = time.time()

    tgem.FBA_res_file_path = fba_output_path
    tgem.concurrent_infer_v(
        v_si=v_si,
        v_ei=v_ei,
        batch_size=batch_size,
        thread=thread
    )

    print(f"FBA finished in {(time.time() - t0) / 60:.2f} min")

