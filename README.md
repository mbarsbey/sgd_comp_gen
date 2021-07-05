# Repository for the paper _Heavy Tails in SGD and Compressibility of Overparametrized Neural Networks_

This repository includes the accompanying code for the paper https://arxiv.org/abs/2106.03795. Here we describe the steps necessary to reproduce the main results presented in the paper. Feel free to contact Melih Barsbey (melih.barsbey@boun.edu.tr) for any questions or comments!

## Training the algorithms

Please use the file `run.py` to specify the combinations of parameters you want to run the experiments for, as well as the folder in which you want them to be saved. The experiment jobs will be written to files inside the folder `./jobs/`. Run these files to produce the results in your desired results directory.

### Warnings

- Make sure you allocate a separate folder for every architecture-dataset combination! 
- Please see the file `all_experiment_settings.csv` for all models presented in the paper.
- Set all seeds to 0 to obtain original results.

## Alpha estimation

Please run the file `alpha_estimation.py` from the command line with the appropriate parameters (see inside the file for help) to obtain alpha estimations for separate layers of your model, as well as aggragate alpha hat values such as mean or median. Training and test accuracies and losses will also be included in this file. The results will be written inside a folder inside your main results folder that includes "results_summary" as a `.csv` file.

## Magnitude pruning

To conduct magnitude pruning, run the file `magnitude_pruning.py` with the appropriate parameters (see inside the file for help). Results of magnitude pruning will be written inside the results summary folder as a `.csv` file.

## Singular value pruning

To conduct singular value pruning, first run the file `svd.py` with the appropriate parameters. This will create and store the results of the SVD inside your results folder. Then run the file `spectral_pruning.py` with the appropriate parameters (see inside the file for help). Results will be written inside the results summary folder as a `.csv` file.

## Node pruning

To conduct node pruning, run the file `nb_pruning.py` with the appropriate parameters (see inside the file for help). Results of magnitude pruning will be written inside the results summary folder as a `.csv` file.

## Other experiments

For other numerical results presented in the paper see the `other_experiments.ipynb` file.
