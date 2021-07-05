from exp_utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rf", help="results root folder", type=str)
args_cl = parser.parse_args()

results_folder = ResultsFolder(args_cl.rf)
_ = alpha_estimation(results_folder)