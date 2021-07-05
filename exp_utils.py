import torch, pandas as pd, numpy as np, math, levy, powerlaw, scipy, scipy.io as sio; #from numba import jit #from numpy import savez_compressed
import pickle, sys, logging, re, os, json, warnings, matplotlib.colors as mcolors, matplotlib.cm as cm, matplotlib.pyplot as plt, matplotlib
from joblib import Parallel, delayed; from numpy.linalg import solve; from spgl1 import spg_bp
from scipy.linalg import svd, norm, inv, lstsq; from scipy.special import logsumexp; from scipy.stats import multivariate_normal as mvn; from scipy.stats import levy_stable
from torchvision import datasets, transforms; from datetime import datetime #from scipy.optimize import nnls #from fnnls import fnnls as nnls; print("Using the fast nnls variant") #from sklearn.metrics import ndcg_score #from sgd_ht_utils import get_eta_crit, write_json, read_json, print_if_freq
from scipy.optimize import nnls
from scipy.stats import percentileofscore
from levy import fit_levy

EF = ["20210407_fcn_cifar10", "20210412_larger_fcn_cifar10", "20210419_vgg11_cifar10", "20210422_small_fcn_mnist", "20210424_large_fcn_mnist", "20210420_vgg11_mnist"]

AVAILABLE_PARAMS = (2, 4)
SPECTRAL_PRUNING_RATIOS = list(np.round([0., .25, .5, .6] + list(np.arange(.65, .905, .05)) + list(np.arange(.91, 1.01, .01)), 2))
MAGNITUDE_PRUNING_RATIOS = [0., .1, .2, .3, .4] + list(np.arange(.5, 1.01, .025))
NB_PRUNING_RATIOS = list(np.arange(.0, 1.01, .05))

device = torch.device('cuda')
logger = logging.getLogger(); logger.setLevel(logging.CRITICAL)

### PRUNING

def get_svd(results_folder, x_type, start=0, end=0):
    end = len(results_folder.folders) if end == 0 else end
    root_folder = results_folder.results_root_folder + f"{results_folder.timestamp}_svd_{x_type}/"
    os.mkdir(root_folder) if not os.path.exists(root_folder) else None 
    for folder in results_folder.folders[start:end]:
        print(folder)
        file_name = root_folder + clean_folder(folder) + ".pkl"
        if os.path.exists(file_name):
            print("Results exist, moving on.")
            continue
        result = {}
        model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=True, x_type=x_type)
        with torch.no_grad():
            for pi, p in enumerate(layers):
                with torch.no_grad():
                    a = p.to("cpu").numpy()
                if p.dim() == 4:
                    a = np.reshape(a, (np.prod(a.shape[:1]), np.prod(a.shape[1:])))
                elif p.dim() != 2:
                    raise Exception
                a_median = np.median(a, axis=0)
                a = a - a_median;
                U, s, Vh = scipy.linalg.svd(a.copy())
                result[pi] = {"U":U, "s":s, "Vh":Vh, "a_median":a_median}

        with open(file_name, "wb") as f:
            pickle.dump(result, f)

def nb_pruning(results_folder, pruning_ratios, pruned_layers_str, x_type, write_results=True, results_part_string="", start=0, end=0, adaptive=False):
    #assert results_folder.model_name == "fcn"
    
    end = len(results_folder.folders) if end == 0 else end
    if results_part_string == "":
        assert (start == 0) and (end == len(results_folder.folders)) 
    train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(results_folder.args)
    
    pruned_layers = get_pruned_layers(pruned_layers_str)
    
    folders = results_folder.folders[start:end]
    dfp = pd.DataFrame(index=range(len(folders)*len(pruning_ratios)), columns=["lr", "b", "lr_b", "pruning_ratio", "test_loss", "test_accuracy", "pruned_layers_str", "folder", "remaining_parameter_ratio"])
    for fi, folder in enumerate(folders):#folders:
        print(folder)
        if "FCN" not in results_folder.model_name:
            for pri, pruning_ratio in enumerate(pruning_ratios):
                print(pruning_ratio)
                total_unpruned_parameters = 0
                model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=True, x_type=x_type)
                with torch.no_grad():
                    for pi, p in enumerate(layers):
                        if pi not in pruned_layers:
                            total_unpruned_parameters += np.prod(p.shape)
                            continue
                        a = get_layer_to_numpy(p).copy(); assert len(a.shape) in AVAILABLE_PARAMS
                        if len(p.shape) == 4:
                            a_dims = a.shape
                            a = np.reshape(a, (np.prod(a.shape[:2]), np.prod(a.shape[2:])))
                        a_median = np.median(a, axis=0, keepdims=True) 
                        a = a - a_median
                        norms = np.linalg.norm(a, axis=1)
                        norms = norms - np.median(norms)
                        if adaptive:
                            idx = int((percentileofscore(np.sqrt(np.cumsum(np.sort(np.abs(norms)) ** 2))/np.linalg.norm(norms), pruning_ratio)/100)*len(norms))
                        else:
                            idx = int(pruning_ratio * a.shape[0])
                        a[np.argsort(norms)[:idx], :] = 0.
                        a += a_median
                        total_unpruned_parameters += int(a.shape[0] - idx) * a.shape[1]
                        if len(p.shape) == 4:
                            a = np.reshape(a, (a_dims)) 
                        p.data = torch.Tensor(a)
                te_hist, te_outputs = eval(test_loader_eval, model.to(device), results_folder.crit, args, if_print=False)
                total_remaining_params = sum([np.prod(layer.shape) for layer in model.parameters() if layer.dim() not in AVAILABLE_PARAMS])
                remaining_parameter_ratio = (total_unpruned_parameters + total_remaining_params) / get_total_params(model) 
                dfp.loc[fi*len(pruning_ratios) + pri] = (args.lr, args.batch_size_train, args.lr/args.batch_size_train, pruning_ratio, te_hist[0], te_hist[1], pruned_layers_str, clean_folder(folder), remaining_parameter_ratio)
        else:
            for pri, pruning_ratio in enumerate(pruning_ratios):
                print(pruning_ratio)
                total_unpruned_parameters = 0
                model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=False, x_type=x_type)
                with torch.no_grad():
                    for i in range(len(layers)):
                        if i not in pruned_layers:
                            total_unpruned_parameters += np.prod(p.shape)
                            continue
                        p = layers[i]
                        a = get_layer_to_numpy(p).copy(); assert len(a.shape) in AVAILABLE_PARAMS
                        a_median = np.median(a, axis=1, keepdims=True)
                        a = a - a_median
                        norms = np.linalg.norm(a, axis=0)
                        #r = fit_levy(norms)
                        norms = norms - np.median(norms)
                        if adaptive:
                            idx = int((percentileofscore(np.sqrt(np.cumsum(np.sort(np.abs(norms)) ** 2))/np.linalg.norm(norms), pruning_ratio)/100)*len(norms))
                        else:
                            idx = int(pruning_ratio * a.shape[1])
                        a[:, np.argsort(norms)[:idx]] = 0.
                        
                        a += a_median
                        
                        total_unpruned_parameters += int(a.shape[1] - idx) * a.shape[0]
                        
                        if len(p.shape) == 4:
                            a = np.reshape(a, (a_dims)) 
                        p.data = torch.Tensor(a)
                te_hist, te_outputs = eval(test_loader_eval, model.to(device), results_folder.crit, args, if_print=False)
                total_remaining_params = sum([np.prod(layer.shape) for layer in model.parameters() if layer.dim() not in AVAILABLE_PARAMS])
                remaining_parameter_ratio = (total_unpruned_parameters + total_remaining_params) / get_total_params(model) 
                dfp.loc[fi*len(pruning_ratios) + pri] = (args.lr, args.batch_size_train, args.lr/args.batch_size_train, pruning_ratio, te_hist[0], te_hist[1], pruned_layers_str, clean_folder(folder), remaining_parameter_ratio)        
        
        
    dfp["pruned_parameter_ratio"] = 1 - dfp["remaining_parameter_ratio"] 
    if write_results:
        dfp.to_csv(results_folder.results_summary_folder + f"{results_folder.timestamp}_{pruned_layers_str}_{x_type}_{'' if adaptive else 'non'}adaptive_nb_pruning{results_part_string}.csv", index=False)
    return dfp
            
def spectral_pruning(results_folder, pruning_ratios, pruned_layers_str, x_type, write_results=True, results_part_string="", start=0, end=0, adaptive=False):
    end = len(results_folder.folders) if end == 0 else end
    if results_part_string == "":
        assert (start == 0) and (end == len(results_folder.folders)) 
    train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(results_folder.args)
    
    pruned_layers = get_pruned_layers(pruned_layers_str)

    folders = results_folder.folders[start:end]
    dfp = pd.DataFrame(index=range(len(folders)*len(pruning_ratios)), columns=["lr", "b", "lr_b", "pruning_ratio", "test_loss", "test_accuracy", "pruned_layers_str", "folder", "remaining_parameter_ratio"])
    for fi, folder in enumerate(folders):#folders:
        print(folder)
        try:
            comp = pickle_load(results_folder.results_root_folder + f"{results_folder.timestamp}_svd_{x_type}/{clean_folder(folder)}.pkl")
        except:
            print("SVD results do not exist")
            continue
        for pri, pruning_ratio in enumerate(pruning_ratios):
            print(pruning_ratio)
            total_unpruned_parameters = 0
            model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=True, x_type=x_type)
            with torch.no_grad():
                for pi, p in enumerate(layers):
                    if pi not in pruned_layers:
                        total_unpruned_parameters += np.prod(p.shape)
                        continue
                    U, s, Vh = comp[pi]["U"], comp[pi]["s"].copy(), comp[pi]["Vh"]
                    a_median = comp[pi]["a_median"]
                    idx = len(s) - int(len(s)*pruning_ratio)
                    s[idx:] = 0
                    total_unpruned_parameters += idx*(U.shape[0]+Vh.shape[0])
                    if p.dim() == 4:
                        p.data = torch.Tensor(np.reshape(U @ scipy.linalg.diagsvd(s, U.shape[0], Vh.shape[0]) @ Vh + a_median, (p.shape[0], p.shape[1],p.shape[2],p.shape[3])))
                    elif p.dim() == 2:
                        p.data = torch.Tensor(U @ scipy.linalg.diagsvd(s, U.shape[0], Vh.shape[0]) @ Vh + a_median)
            te_hist, te_outputs = eval(test_loader_eval, model.to(device), results_folder.crit, args, if_print=False)
            total_remaining_params = sum([np.prod(layer.shape) for layer in model.parameters() if layer.dim() not in AVAILABLE_PARAMS])
            remaining_parameter_ratio = (total_unpruned_parameters + total_remaining_params) / get_total_params(model) 
            dfp.loc[fi*len(pruning_ratios) + pri] = (args.lr, args.batch_size_train, args.lr/args.batch_size_train, pruning_ratio, te_hist[0], te_hist[1], pruned_layers_str, clean_folder(folder), remaining_parameter_ratio)
    dfp["pruned_parameter_ratio"] = 1 - dfp["remaining_parameter_ratio"] 

    dfp.to_csv(results_folder.results_summary_folder + f"{results_folder.timestamp}_{pruned_layers_str}_{x_type}_{'' if adaptive else 'non'}adaptive_spectral_pruning{results_part_string}.csv", index=False)

def magnitude_pruning(results_folder, pruning_ratios, pruned_layers_str, x_type, write_results=True, results_part_string="", start=0, end=0, adaptive=False, global_pruning=False):
    end = len(results_folder.folders) if end == 0 else end
    if results_part_string == "":
        assert (start == 0) and (end == len(results_folder.folders)) 
    train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(results_folder.args)
    
    pruned_layers = get_pruned_layers(pruned_layers_str)
    
    folders = results_folder.folders[start:end]
    dfp = pd.DataFrame(index=range(len(folders)*len(pruning_ratios)), columns=["lr", "b", "lr_b", "pruning_ratio", "test_loss", "test_accuracy", "pruned_layers_str", "folder", "remaining_parameter_ratio"])
    for fi, folder in enumerate(folders):#folders:
        print(folder)
        
        for pri, pruning_ratio in enumerate(pruning_ratios):
            print(pruning_ratio)
            total_unpruned_parameters = 0
            model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=True, x_type=x_type)
            
            if global_pruning:
                all_p_flat = np.array([])
                with torch.no_grad():
                    for pi, p in enumerate(layers):
                        if pi not in pruned_layers:
                            total_unpruned_parameters += np.prod(p.shape)
                            continue
                        p_flat = p.flatten()
                        p_median = np.median(p_flat)
                        all_p_flat = np.concatenate((all_p_flat, p_flat - p_median))
                    if adaptive:
                        idx = int((percentileofscore(np.sqrt(np.cumsum(np.sort(np.abs(all_p_flat)) ** 2))/np.linalg.norm(all_p_flat), pruning_ratio)/100)*len(all_p_flat))
                    else:
                        idx = int(pruning_ratio * len(all_p_flat))
                    all_p_flat[np.argsort(np.abs(all_p_flat))[:idx]] = 0.
                total_unpruned_parameters += len(all_p_flat[idx:]) 
                cur_idx = 0
                with torch.no_grad():
                    for pi, p in enumerate(layers):
                        if pi not in pruned_layers:
                            continue #(let's not count parameters twice)
                        p_flat = p.flatten()
                        p_median = np.median(p_flat)
                        p_flat = all_p_flat[cur_idx:cur_idx + len(p_flat)] + p_median
                        p.data = torch.reshape(torch.Tensor(p_flat), p.shape)

                        cur_idx = cur_idx + len(p_flat)
                    
            else:
                with torch.no_grad():
                    for pi, p in enumerate(layers):
                        if pi not in pruned_layers:
                            total_unpruned_parameters += np.prod(p.shape)
                            continue
                        p_flat = p.flatten()
                        p_median = np.median(p_flat)
                        p_flat -= p_median
                        if adaptive:
                            idx = int((percentileofscore(np.sqrt(np.cumsum(np.sort(np.abs(p_flat)) ** 2))/np.linalg.norm(p_flat), pruning_ratio)/100)*len(p_flat))
                        else:
                            idx = int(pruning_ratio * len(p_flat))
                        p_flat[torch.argsort(torch.abs(p_flat))[:idx]] = 0.

                        total_unpruned_parameters += len(p_flat[idx:]) 

                        p_flat += p_median
                        p.data = torch.reshape(p_flat, p.shape)
                    
            te_hist, te_outputs = eval(test_loader_eval, model.to(device), results_folder.crit, args, if_print=False)
            total_remaining_params = sum([np.prod(layer.shape) for layer in model.parameters() if layer.dim() not in AVAILABLE_PARAMS])
            remaining_parameter_ratio = (total_unpruned_parameters + total_remaining_params) / get_total_params(model) 
            dfp.loc[fi*len(pruning_ratios) + pri] = (args.lr, args.batch_size_train, args.lr/args.batch_size_train, pruning_ratio, te_hist[0], te_hist[1], pruned_layers_str, clean_folder(folder), remaining_parameter_ratio)
    dfp["pruned_parameter_ratio"] = 1 - dfp["remaining_parameter_ratio"] 
    if write_results:
        dfp.to_csv(results_folder.results_summary_folder + f"{results_folder.timestamp}_{pruned_layers_str}_{x_type}_{'global' if global_pruning else 'local'}_{'' if adaptive else 'non'}adaptive_magnitude_pruning{results_part_string}.csv", index=False) # {'_bug' if adaptive_bug else ''}
    return dfp 


### RESULTS FOLDER PROCESSING

class ResultsFolder(object):
    def __init__(self,results_root_folder):
        self.results_root_folder = results_root_folder
        self.timestamp, self.folders, self.results_summary_folder = prepare_result_analysis(results_root_folder)
        model, self.args, self.model_name, self.dataset_name, _, self.no_layers = get_model_info(self.folders[0], dream_team=True, x_type="x_final")
        if self.args.criterion == 'NLL':
            self.crit = torch.nn.CrossEntropyLoss(reduction='mean').to(self.args.device)    
        self.param_sizes = []
        with torch.no_grad():
            for pi, p in enumerate(model.parameters()):
                self.param_sizes.append(p.shape)
        self.total_params = [np.prod(size) for size in self.param_sizes]

def prepare_result_analysis(results_root_folder):
    assert (results_root_folder[-1] == "/") and (results_root_folder != "results/")
    timestamp = results_root_folder.split("/")[1].split("_")[0];
    timestamp = timestamp if timestamp.isnumeric() else "00000000"
    folders = sorted([r for r in [results_root_folder + f + "/" for f in os.listdir(results_root_folder)] if os.path.isdir(r) and os.path.exists(r + "net.pyT")])
    results_summary_folder = results_root_folder + f"{timestamp}_results_summary/"
    os.makedirs(results_summary_folder) if not os.path.exists(results_summary_folder) else None
    return timestamp, folders, results_summary_folder

def get_model_info(m, dream_team, x_type):
    model_file_name = "avg_net.pyT" if x_type == "x_mc" else "net.pyT"
    model = torch.load(m + model_file_name,map_location='cpu')
    args = torch.load(m + "args.info",map_location='cpu')
    model_name = args.model.upper()
    dataset_name = args.dataset.upper()
    layers = get_layers(model, dream_team=dream_team)
    no_layers = len(layers)
    return model, args, model_name, dataset_name, layers, no_layers

### TAIL INDEX ESTIMATION

def alpha_estimation(results_folder, x_type="x_mc", write_results=True, results_part_string="", start=0, end=0):
    end = len(results_folder.folders) if end == 0 else end
    if results_part_string == "":
        assert (start == 0) and (end == len(results_folder.folders))
    layer_cols = [[f"alpha_hat_layer_{i}", f"alpha_hat_norms_layer_{i}"] for i in list(range(results_folder.no_layers))] #keeping this for alpha estimation experiments
    all_layer_cols = [j for i in layer_cols for j in i]; all_layer_cols_ind = [f"alpha_hat_ind_layer_{i}" for i in list(range(results_folder.no_layers))] 
    cols = ["batch_size", "learning_rate", "width"] + all_layer_cols + all_layer_cols_ind + ["training_error", "test_error", "error_diff"] + ["training_accuracy", "test_accuracy", "accuracy_diff", "lr_b","training_accuracy_avg", "test_accuracy_avg", "accuracy_diff_avg", "folder"]
    folders = results_folder.folders[start:end]
    df = pd.DataFrame(index=range(len(folders)), columns=cols)
    for fi, folder in enumerate(folders):
        try:
            print(folder); assert torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][1] < 5e-5 # check for convergence
            model, args, model_name, dataset_name, layers, no_layers = get_model_info(folder, dream_team=True, x_type=x_type)
            for pi, p in enumerate(layers):
                a = get_layer_to_numpy(p).copy(); assert len(a.shape) in AVAILABLE_PARAMS
                a = np.reshape(a, (np.prod(a.shape[:2]), np.prod(a.shape[2:]))) if len(a.shape) == 4 else a
                a = a - np.median(a, axis=0) 
                norms = np.linalg.norm(a, axis=1)
                #r =  fit_levy(norms)
                #norms = norms - r[0].get()[2]
                norms = norms - np.median(norms)
                a_ind = a.flatten() - np.median(a); assert len(a_ind) == np.prod(a.shape)
                df.loc[fi, layer_cols[pi]]              = np.median([est_alpha(np.random.permutation(a)) for i in range(5)])
                df.loc[fi, f"alpha_hat_ind_layer_{pi}"] = np.median([est_alpha_one(np.random.permutation(a_ind)) for i in range(5)])     # permuting the independent vectors
                df.loc[fi, [f"alpha_hat_ind_alt{i}_layer_{pi}" for i in range(1,6)]] = est_alpha_one_alternatives(a_ind)
                df.loc[fi, f"alpha_hat_norms_layer_{pi}"] = np.median([est_alpha_one(np.random.permutation(norms)) for i in range(5)])     
                df.loc[fi, [f"alpha_hat_norms_alt{i}_layer_{pi}" for i in range(1,6)]] = est_alpha_one_alternatives(norms)
            df.loc[fi, ["batch_size", "learning_rate", "lr_b", "width", "folder"]] = args.batch_size_train, args.lr, args.lr/ args.batch_size_train , args.width, clean_folder(folder)
            _ = get_loss_and_accuracy(df, fi, folder)
        except FileNotFoundError:
            print("File not found!")
    df_alpha_hats     = df[[f for f in all_layer_cols     if ("half" not in f) and ("norms" not in f)]]
    df_alpha_hats_ind = df[[f for f in all_layer_cols_ind if ("half" not in f)]]
    df_alpha_hats_norms = df[[f for f in all_layer_cols_ind if ("half" not in f)and ("norms" in f)]]
    df["alpha_hat_median"], df["alpha_hat_median_ind"], df["alpha_hat_median_norms"] = df_alpha_hats.median(1), df_alpha_hats_ind.median(1), df_alpha_hats_norms.median(1)
    df["alpha_hat_mean"],   df["alpha_hat_mean_ind"], df["alpha_hat_mean_norms"]   = df_alpha_hats.mean(1),   df_alpha_hats_ind.mean(1), df_alpha_hats_norms.mean(1)
    df["alpha_hat_min"],    df["alpha_hat_min_ind"], df["alpha_hat_min_norms"]    = df_alpha_hats.min(1),    df_alpha_hats_ind.min(1), df_alpha_hats_norms.min(1)

    if write_results:
        df.to_csv(results_folder.results_summary_folder + results_folder.timestamp + f"_{x_type}_results{results_part_string}.csv", index=False)
    return df

def simulate_mv_stable(d , n, alp, elliptic=True):
    if(elliptic):
        phi = levy_stable.rvs(alp/2, 1, loc=0, scale=2*np.cos(np.pi*alp/4)**(2/alp), size=(1,n))
        mv_stable = np.sqrt(phi) * np.random.randn(d,int(n))
    else:
        mv_stable = levy_stable.rvs(alpha = alp, beta = 0, scale = 1, size=(d,n))
    
    return mv_stable

def alpha_estimator_one(m, X, estimator):
    N = len(X)
    n = int(N/m) # must be an integer
    eps = np.spacing(1)
    if estimator == "mmo2014":    
        Y = torch.sum(X.reshape(n, m),1)
        
        Y_log_norm =  np.log(np.abs(Y) + eps).mean()
        X_log_norm =  np.log(np.abs(X) + eps).mean()
        diff = (Y_log_norm - X_log_norm) / math.log(m)
        return 1 / diff
    elif estimator in ["hill1975", "dekkers1989", "dehaan1998"]:
        Y = np.sort(np.abs(X))
        log_Y = np.log(Y + eps)
        gamma = (log_Y[-m:] - log_Y[-(m+1)]).mean()
        if estimator == "hill1975":
            return 1/gamma
        elif estimator == "dekkers1989":
            M = np.sum((log_Y[-m:] - log_Y[-(m+1)])**2)/m
            gamma = gamma + 1 - 0.5*(1-(gamma)**2/M)**-1
            return 1/gamma 
        elif estimator == "dehaan1998":
            M = np.sum((log_Y[-m:] - log_Y[-(m+1)])**2)/m
            gamma = M/(2*gamma)
            return 1/gamma
    elif estimator == "pickands1975":
        Y = np.sort(np.abs(X))
        gamma = np.log((Y[-(m//4 + 1)] - Y[-(m//2 + 1)])/(Y[-(m//2 + 1)] - Y[-(m + 1)]))/np.log(2)
        return 1/gamma
    elif estimator == "paulauskas2011":    
        Y = np.sort(np.abs(X.reshape(n, m)), axis=1)
        Z = (Y[:, -2]/Y[:, -1]).sum()/n
        return Z / (1 - Z)
    else:
        raise NotImplementedError
    
def alpha_estimator_multi(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff.item()

def find_m(N, m=0):
    if m == 0:
        m = int(np.sqrt(N)); assert N < 3e7+1
    while N % m != 0:
       m -= 1
    return m

def est_alpha_one_alternatives(X):
    return est_alpha_one(X, estimator="hill1975"), est_alpha_one(X, estimator="pickands1975"), est_alpha_one(X, estimator="dehaan1998"), est_alpha_one(X, estimator="dekkers1989"), est_alpha_one(X, estimator="paulauskas2011")

def est_alpha_one(X, center=False, estimator="mmo2014"):
    assert len(X.shape) == 1
    N = len(X)
    if center:
        X_mean = X.mean()
        X = X - X_mean
    m = find_m(N)
    cands = [c if N % c == 0 else find_m(N, c) for c in [k for k in [2, 5, 10, 20, 50, 100, 500, 1000] if k <= N/2]] if N != 10 else [2,5];
    assert 1 not in cands
    cands = sorted(list(set(cands)))
    if (m not in cands) and (m>1):
        cands.append(m)
    alp_tmp = [alpha_estimator_one(mm, torch.from_numpy(X), estimator=estimator) for mm in cands] 
    return min(np.median(alp_tmp), 2.0)

def est_alpha(X, center=False):
    N, d = X.shape; #assert N >= d
    if center:
        X_mean = X.mean(0); assert len(X_mean) == d
        X = X - X_mean
    m = find_m(N)
    cands = [c if N % c == 0 else find_m(N, c) for c in [k for k in [2, 5, 10, 20, 50, 100, 500, 1000] if k <= N/2]] if N != 10 else [2,5] 
    assert 1 not in cands
    cands = sorted(list(set(cands)))
    if (m not in cands) and (m>1):
        cands.append(m)
    alp_tmp = [alpha_estimator_multi(mm, torch.from_numpy(X)) for mm in cands] 
    return min(np.median(alp_tmp), 2.0)

### MISC
    
def get_layers(model, dream_team):
    layers = [layer for layer in model.parameters() if (layer.dim() >= 2)]
    if dream_team:
        layers[-1] = layers[-1].T
    return layers

def get_pruned_layers(pruned_layers_str):
    assert "10" not in pruned_layers_str
    return [int(i) for i in pruned_layers_str]

def get_total_params(model):
    return sum([np.prod(layer.shape) for layer in model.parameters()])

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        ret = pickle.load(f)
    return ret 

def clean_folder(folder):
    #return folder.replace("results/","").replace("/", "")
    return folder.split("/")[-2]

def get_layer_to_numpy(p):
    with torch.no_grad():
        a = p.to("cpu").numpy()
    return a

def get_folder_idx(orig_folder_length, k, T):
    no_exps = orig_folder_length // T
    start = (k-1) * no_exps
    end = k  * no_exps if k < T else orig_folder_length
    return no_exps, start, end

def get_results_part_string(eparallel, no_samples):
    results_part_string = "" if eparallel == "_1_1" else "_part" + eparallel
    if results_part_string == "":
        if no_samples != -1:
            results_part_string = f"_{no_samples}_samples"
    return results_part_string
def sample_result_folders(results_folder, no_samples):
    df = pd.read_csv(results_folder.results_summary_folder + results_folder.timestamp + "_x_mc_results.csv")
    df = df.sort_values("lr_b")
    chosen_folders = df.folder.iloc[list(map(int, np.linspace(0, len(df.folder)-1, no_samples)))].to_list()
    return [f for f in results_folder.folders if any([k + "/" in f for k in chosen_folders])]

### DATA

def get_data(args):

    # mean/std stats
    if args.dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, 0.243, 0.262]
            }
    elif args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.5071, 0.4867, 0.4408] ,
            'std': [0.2675, 0.2565, 0.2761]
            }
    elif args.dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        input_dim = 28 * 28
        stats = {
            'mean': [0.1307],
            'std': [0.3081]
            }
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
        ]

    # get tr and te data with the same normalization
    # no preprocessing for now
    tr_data = getattr(datasets, data_class)(
        root=args.path,
        train=True,
        download=True,
        transform=transforms.Compose(trans)
        )

    te_data = getattr(datasets, data_class)(
        root=args.path,
        train=False,
        download=True,
        transform=transforms.Compose(trans)
        )

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
        )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )
    return train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim


### EVALUATION

def get_loss_and_accuracy(df, fi, folder):
    df.loc[fi, "training_loss"] = torch.load(folder + "evaluation_history_TRAIN.hist",map_location='cpu')[-1][1]
    df.loc[fi, "test_loss"] = torch.load(folder + "evaluation_history_TEST.hist",map_location='cpu')[-1][1]
    df.loc[fi, "loss_diff"] = np.abs(df.loc[fi, "test_loss"] - df.loc[fi, "training_loss"])
    df.loc[fi, "training_accuracy"] = torch.load(folder + "evaluation_history_TRAIN.hist",map_location='cpu')[-1][2]
    df.loc[fi, "test_accuracy"] =     torch.load(folder + "evaluation_history_TEST.hist",map_location='cpu')[-1][2]
    df.loc[fi, "accuracy_diff"] = np.abs(df.loc[fi, "test_accuracy"] - df.loc[fi, "training_accuracy"])
    df.loc[fi, "training_loss_avg"] = torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][1]
    df.loc[fi, "test_loss_avg"] = torch.load(folder + "evaluation_history_AVG.hist",map_location='cpu')[-1][1]
    df.loc[fi, "loss_diff_avg"] = np.abs(df.loc[fi, "test_loss_avg"] - df.loc[fi, "training_loss_avg"])
    df.loc[fi, "training_accuracy_avg"] = torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][2]
    df.loc[fi, "test_accuracy_avg"] =     torch.load(folder + "evaluation_history_AVG.hist",map_location='cpu')[-1][2]
    df.loc[fi, "accuracy_diff_avg"] = np.abs(df.loc[fi, "test_accuracy_avg"] - df.loc[fi, "training_accuracy_avg"])
    
def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)

def eval(eval_loader, net, crit, args, if_print=True):

    net.eval()
    # run over both test and train set
    total_size = 0
    total_loss = 0
    total_acc = 0
    outputs = []

    with torch.no_grad():
        P = 0  # num samples / batch size
        for x, y in eval_loader:
            P += 1
            # loop over dataset
            x, y = x.to(args.device), y.to(args.device)
            out = net(x)
            outputs.append(out)

            loss = crit(out, y)
            prec = accuracy(out, y)
            bs = x.size(0)

            total_size += int(bs)
            total_loss += float(loss) * bs
            total_acc += float(prec) * bs

        hist = [total_loss / total_size, total_acc / total_size]
        if if_print:
            print(hist)

        return hist, outputs
    
### TRAINING NEURAL NETWORKS
    
def update_avg_net(net, avg_net, num_iter, burn_in=1000):
    n = num_iter - burn_in + 1
    if num_iter < burn_in: # corrected this from <= to <
        return avg_net #changed this to use less memory #copy.deepcopy(net)
    else:
        with torch.no_grad():
            for (p_avg, p_new) in zip(avg_net.parameters(), net.parameters()):
                p_avg.data = (1 - 1 / n) * p_avg.data + (1 / n) * p_new.data
            return avg_net
