import os
import subprocess
import itertools
import time
import math
import copy
import numpy as np
# folder to save
base_path = "results/ongoing_experiments"
if not os.path.exists(base_path):
    os.makedirs(base_path)

# experimental setup
width = [2048]
depth = [5]
seeds = [0]  
dataset = ['mnist']  # choose one of 'mnist' or 'cifar10'
models = ['fcn'] # please choose one of 'fcn' (for fully connected exps), 'vgg11' (for VGG11-like CNN for CIFAR10 dataset), or 'vgg11_bw' (for VGG11-like CNN for MNIST dataset)
loss = ['NLL']  
betas = [0.] 
batch_sizes = [50]
gamma0s = [.1, .15, .2] # this corresponds to learning rates for this experiment
num_iter = 1000000
num_iter_mc = 1000

no_files = 4

alphas = [0.0]
record_extra_iters = False
pbs_file_stem = "" # keep this empty if you do not want to run this on SGE

grid = itertools.product(width, depth, seeds, models, dataset, loss, betas, alphas, batch_sizes, gamma0s)
no_exps = len(list(copy.deepcopy(grid)))
processes = []

if no_files > 1:
    [os.remove("jobs/"+file_name) for file_name in os.listdir("jobs/") if "jobs" in file_name]
jobs_file = open('jobs/jobs.sh', 'w')

security = """\n"""
jobs_file.write(security)

               
for w, h, s, model, d, l, beta, alpha, batch_size, gamma0 in grid:
    save_dir = base_path + '/{:05d}_{}_{:02d}_{}_{}_{}_{}_{}_{}_{}'.format(w, h, s, model, d, l, beta, 1.0 - alpha, batch_size, gamma0)
    lr = gamma0 * (w ** beta) # step-size
    #num_iter = T * (w / lr) ** (1 /(1 - alpha)) # number of iterations has been moved to the start of the file as a free parameter 
    if os.path.exists(save_dir) and not record_extra_iters:
       print('folder already exists, omitting the current setting') # folder created only at the end when all is done!
       no_exps -= 1
       continue
    cmd = 'python main.py '
    cmd += '--save_dir {} '.format(save_dir)
    cmd += '--width {} '.format(w)
    cmd += '--depth {} '.format(h)
    cmd += '--seed {} '.format(s)
    cmd += '--model {} '.format(model)
    cmd += '--dataset {} '.format(d)
    cmd += '--lr {} '.format(lr)
    cmd += '--iterations {} '.format(int(num_iter))
    cmd += '--batch_size_train {} '.format(batch_size)
    cmd += '--alpha {} '.format(alpha) #cmd += '--custom_init '
    cmd += '--mc_iterations {} '.format(num_iter_mc) # last k iterations for mc average
    if record_extra_iters:
        cmd+= '--record_extra_iters'
        jobs_file.write('{:s}; \n'.format(cmd))
    else:
        cmd += '--traj ' # added the traj parameter, to keep the monte carlo average
        log_file = save_dir + '.log' 
        jobs_file.write('{:s} >> {:s} 2>&1 ;\n'.format(cmd, log_file))
jobs_file.close()



pbs_prefix = """\n"""


if no_files > 1:
    import numpy as np
    with open("jobs/jobs.sh", "r") as jobs_file:
        all_jobs = jobs_file.read()
    all_jobs = all_jobs.split("\n")[2:-1]; assert len(all_jobs) == no_exps
    sep_jobs = np.array_split(all_jobs, no_files)
    if pbs_file_stem != "":
	pass
    else:
        for i in range(no_files):
            with open(f"jobs/jobs{str(i+1).rjust(2,'0')}.sh", "w") as jobs_file:
                jobs_file.write(security)
                jobs_file.write("\n".join(sep_jobs[i]))
        os.remove('jobs/jobs.sh')
