from exp_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rf", help="results root folder", type=str)
parser.add_argument("--x_type", help="x_type", type=str)
parser.add_argument("--no_samples", help="take regular samples based on lr/b", type=int, default=-1)
parser.add_argument("--eparallel", help="implementation of embarrassingly parallellness", type=str, default="_1_1")
args_cl = parser.parse_args()

assert args_cl.eparallel[0] == "_"; assert (args_cl.rf[-1] == "/") and (args_cl.rf != "results/")

k, T = list(map(int, args_cl.eparallel[1:].split("_")))
results_folder = ResultsFolder(args_cl.rf)
if args_cl.no_samples != -1:
    results_folder.folders = sample_result_folders(results_folders=results_folder, no_samples=args_cl.no_samples)
no_exps, start, end = get_folder_idx(orig_folder_length=len(results_folder.folders), k=k, T=T)
results_part_string = get_results_part_string(args_cl.eparallel, args_cl.no_samples)
    
get_svd(results_folder, x_type=args_cl.x_type, start=start, end=end)
        

