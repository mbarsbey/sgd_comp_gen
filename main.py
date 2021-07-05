import argparse
import datetime
import math
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.parallel
import vgg
import sys
import numpy as np

from models import MultiLayerNN
from exp_utils import get_data, accuracy, eval, update_avg_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=100, type=int,
                        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--print_freq', default=200, type=int)
    parser.add_argument('--eval_freq', default=200, type=int)
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='mnist | cifar10 | cifar100')
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='fcn', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
                        help='NLL | linear_hinge')
    parser.add_argument('--width', default=100, type=int,
                        help='width of fully connected layers')
    parser.add_argument('--depth', default=2, type=int,
                        help='total number of hidden layers + input layer')
    parser.add_argument('--mc_iterations', default=10000, type=int,
                        help='num. of final iterations to compute parameter averages') #added this
    parser.add_argument('--save_dir', default='results/', type=str)
    parser.add_argument('--custom_init', action='store_true', default=False)
    parser.add_argument('--traj', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--schedule', action='store_true', default=False)
    parser.add_argument('--preprocess', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    parser.add_argument('--continue_prev', action='store_true', default=True)
    parser.add_argument('--record_extra_iters', action='store_true', default=False)

    args = parser.parse_args() #parsing arguments
    print(args)
    
    torch.manual_seed(args.seed) # setting seed
    
    burn_in_iterations = args.iterations - args.mc_iterations # number of burn in iterations before mc_iterations are set
    
    begin_time = time.time() #timer begun
    
    if (args.continue_prev == False) or (args.save_dir.split("/")[-1] not in os.listdir("/".join(args.save_dir.split("/")[:-1]))):
        # we create the setup anew if the experiment folder does not exist
        if args.double:
            torch.set_default_tensor_type('torch.DoubleTensor')
        args.use_cuda = not args.no_cuda and torch.cuda.is_available()
        args.device = torch.device('cuda' if args.use_cuda else 'cpu') # use CUDA

        train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(args) #done
        
        if "vgg" in args.model: # TODO
            net = vgg.__dict__[args.model]().to(args.device)
            net.features = torch.nn.DataParallel(net.features)
        else:
            def init_weights(m):
                if type(m) == nn.Linear:
                    m.weight.data.fill_(0.01)
            net = MultiLayerNN(input_dim=input_dim, width=args.width ,depth=args.depth, num_classes=num_classes).to(args.device)
            if args.custom_init:
                net.apply(init_weights)

        file_traj = args.save_dir + '_traj.log'
        f = open(file_traj, 'w+')
        f.write(str(args))
        f.write(str(net))
        f.write("\n")
        f.close()

        avg_net = copy.deepcopy(net) 

        print(net)

        opt = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd
        )

        if args.criterion == 'NLL':
            crit = nn.CrossEntropyLoss(reduction='mean').to(args.device)

        def cycle_loader(dataloader):
            while 1:
                for data in dataloader:
                    yield data

        circ_train_loader = cycle_loader(train_loader)

        # training logs per iteration
        training_history = []
        weight_grad_history = []

        # eval logs less frequently
        evaluation_history_TEST = []
        evaluation_history_TRAIN = []
        evaluation_history_AVG = []
        evaluation_history_AVGTRAIN = []

        STOP = False
        prev_iters = 0
    else:

        folder = args.save_dir + "/"
        iters = args.iterations
        mc_iters = args.mc_iterations
        try:
            record_extra_iters = args.record_extra_iters
        except:
            record_extra_iters = False
        args = torch.load(folder + "args.info")
        args.record_extra_iters = record_extra_iters
        #prev_iters = args.iterations
        if args.record_extra_iters:
            prev_iters = 0
            mc_iters = 1
        else:
            prev_iters = torch.load(folder + "training_history.hist")[-1][0]
        args.iterations = iters + prev_iters
        args.mc_iterations = mc_iters 

        burn_in_iterations = args.iterations - args.mc_iterations
        
        if args.double:
            torch.set_default_tensor_type('torch.DoubleTensor')
        args.use_cuda = not args.no_cuda and torch.cuda.is_available()
        args.device = torch.device('cuda' if args.use_cuda else 'cpu')
        train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(args)

        net = torch.load(folder + "net.pyT")
        avg_net = torch.load(folder + "avg_net.pyT") # changed this, made this the deepcopy

        if args.record_extra_iters:
            extra_iters_folder = folder + "extra_iters/"
            if os.path.exists(extra_iters_folder):
                msg = f"\nCancelling the extra iters experiments since extra iters folder exists.\n\n"
                print(msg)
                sys.exit()
            else:
                os.makedirs(extra_iters_folder)
            evaluation_history_TEST = []
            evaluation_history_TRAIN = []
            evaluation_history_AVG = []
            evaluation_history_AVGTRAIN = []
        else:
            assert args.iterations > args.mc_iterations

            file_traj = args.save_dir + '_traj.log'
            f = open(file_traj, 'a')
            msg = f"\nContinuing from a previous saved model, training for at most {iters} more iterations\n\n"
            f.write(msg)
            f.close()
            print(msg)
            
            # eval logs less frequently
            evaluation_history_TEST = torch.load(folder + "evaluation_history_TEST.hist")
            evaluation_history_TRAIN = torch.load(folder + "evaluation_history_TRAIN.hist")
            evaluation_history_AVG = torch.load(folder + "evaluation_history_AVG.hist")
            evaluation_history_AVGTRAIN = torch.load(folder + "evaluation_history_AVGTRAIN.hist")


        opt = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.mom,
            weight_decay=args.wd
        )

        if args.criterion == 'NLL':
            crit = nn.CrossEntropyLoss(reduction='mean').to(args.device)

        def cycle_loader(dataloader):
            while 1:
                for data in dataloader:
                    yield data

        circ_train_loader = cycle_loader(train_loader)

        training_history = []



        STOP = False
        #       print("finished starting preparing for the model")
            
       
    convergence_first_observed_at = 0
    mc_convergence_coefficient = 10 if args.mc_iterations <= 1000 else 3
    
    for j, (x, y) in enumerate(circ_train_loader):
        if args.record_extra_iters:
            i = j + prev_iters # override keeping the seed the same for new data if recording extra iters
        else:
            i = j
        if i < prev_iters:
            continue
        if args.record_extra_iters:
            torch.save(net, extra_iters_folder + f'net_{i - prev_iters}.pyT')
            tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args, if_print=False)
            te_hist, te_outputs = eval(test_loader_eval, net, crit, args, if_print=False)
            evaluation_history_TRAIN.append([i, *tr_hist])
            evaluation_history_TEST.append([i, *te_hist])
        elif (i % args.eval_freq == 0): #or (j == 0):
            print("## Iteration", i)
            # first record is at the initial point
            print('train eval')
            tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)
            print('test eval')
            te_hist, te_outputs = eval(test_loader_eval, net, crit, args)
            print(tr_hist)
            evaluation_history_TRAIN.append([i, *tr_hist])
            evaluation_history_TEST.append([i, *te_hist])
            if args.traj:
                print('train eval avg_net')
                tat_hist, tat_outputs = eval(train_loader_eval, avg_net, crit, args)
                print('test eval avg_net')
                ta_hist, ta_outputs = eval(test_loader_eval, avg_net, crit, args)
                evaluation_history_AVGTRAIN.append([i, *tat_hist])
                evaluation_history_AVG.append([i, *ta_hist])

            #use traj file
            if args.traj:
                f = open(file_traj, 'a+')
                f.write('## Iteration {:d} \n'.format(i))
                f.write('Training set:\n')
                f.write(str(tr_hist) + '\n')
                f.write('Test set:\n')
                f.write(str(te_hist) + '\n')
                f.write('Avg train set: \n')
                f.write(str(tat_hist) + '\n')
                f.write('Avg test set: \n')
                f.write(str(ta_hist) + '\n')
                f.write('lr: ' + str(opt.param_groups[0]['lr']) + '\n')
                f.write('\n')
                f.close()

        net.train()

        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        # calculate the gradients
        loss.backward()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])

        if args.alpha > 0:
            for group in opt.param_groups:
                gan = (args.lr / args.width) ** (1 / (1 - args.alpha))
                group['lr'] = args.lr * (i + (1 / gan)) ** (- args.alpha)

        opt.step()
        
        # compute mean of the network over the trajectories 
        if args.traj: 
            avg_net = update_avg_net(net, avg_net, i, burn_in_iterations)
            
        if args.lr_schedule:
            scheduler.step(i)

        if i == args.iterations: 
            STOP = True
        
        if not args.record_extra_iters:
            if np.isnan(tr_hist[0]):
                f = open(file_traj, 'a+')
                f.write('Training terminated due to divergence.\n')
                f.close()
                print('Training terminated due to divergence.\n')
                sys.exit()
            
            if (i > 50000) and (tr_hist[1] <= 10.) and (args.dataset in ["cifar10", "mnist"]): #HACK
                f = open(file_traj, 'a+')
                f.write('Training terminated due to divergence.\n')
                f.close()
                print('Training terminated due to divergence.\n')
                sys.exit()

            if (tr_hist[0] < 5e-5) and (tr_hist[1] == 100.):
                if convergence_first_observed_at == 0:
                    convergence_first_observed_at = i
                    burn_in_iterations = i + (mc_convergence_coefficient-1) * args.mc_iterations
                elif i - convergence_first_observed_at >= mc_convergence_coefficient * args.mc_iterations:
                    f = open(file_traj, 'a+')
                    f.write('Finishing training due to convergence.\n')
                    f.close()
                    print('Finishing training due to convergence.\n')
                    STOP = True

        if STOP:
            if args.record_extra_iters:
                torch.save(evaluation_history_TEST, extra_iters_folder + 'evaluation_history_extra_iters_TEST.hist')
                torch.save(evaluation_history_TRAIN, extra_iters_folder + 'evaluation_history_extra_iters_TRAIN.hist')
                
            else:
                # final evaluation and saving results
                print("\n## Final evaluation: ")
                print('train eval')
                tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)
                print('test eval')
                te_hist, te_outputs = eval(test_loader_eval, net, crit, args)

                evaluation_history_TRAIN.append([i + 1, *tr_hist])
                evaluation_history_TEST.append([i + 1, *te_hist])

                if args.traj:
                    print('train eval avg_net')
                    tat_hist, tat_outputs = eval(train_loader_eval, avg_net, crit, args)
                    print('test eval avg_net')
                    ta_hist, ta_outputs = eval(test_loader_eval, avg_net, crit, args)

                    evaluation_history_AVGTRAIN.append([i + 1, *tat_hist])
                    evaluation_history_AVG.append([i + 1, *ta_hist])

                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                else:
                    print('Folder already exists, beware of overriding old data!')

                # save the setup
                torch.save(args, args.save_dir + '/args.info')
                # save the outputs
                torch.save(te_outputs, args.save_dir + '/te_outputs.pyT')
                torch.save(tr_outputs, args.save_dir + '/tr_outputs.pyT')

                if args.traj:
                    torch.save(ta_outputs, args.save_dir + '/ta_outputs.pyT')
                    torch.save(evaluation_history_AVG, args.save_dir + '/evaluation_history_AVG.hist')
                    torch.save(tat_outputs, args.save_dir + '/tat_outputs.pyT')
                    torch.save(evaluation_history_AVGTRAIN, args.save_dir + '/evaluation_history_AVGTRAIN.hist')
                    torch.save(avg_net, args.save_dir + '/avg_net.pyT')

                # save the model
                torch.save(net, args.save_dir + '/net.pyT')
                # save the logs
                torch.save(training_history, args.save_dir + '/training_history.hist')
                torch.save(evaluation_history_TEST, args.save_dir + '/evaluation_history_TEST.hist')
                torch.save(evaluation_history_TRAIN, args.save_dir + '/evaluation_history_TRAIN.hist')

                end_time = time.time()
                total_time = end_time - begin_time
                time_secs = str(datetime.timedelta(seconds=total_time))

                #use traj file
                if args.traj:
                    f = open(file_traj, 'a+')
                    f.write('## End \n')
                    f.write('Training set:\n')
                    f.write(str(tr_hist) + '\n')
                    f.write('Test set:\n')
                    f.write(str(te_hist) + '\n')
                    f.write('Avg train set: \n')
                    f.write(str(tat_hist) + '\n')
                    f.write('Avg test set: \n')
                    f.write(str(ta_hist) + '\n')
                    f.write('\nTotal time: ' + time_secs + '\n')
                    f.write('\n')
                    f.close()
                print("\nTotal Time: " + time_secs)
            break
