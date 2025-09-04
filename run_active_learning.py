import pickle
import os

import copy 

import argparse
import json

import numpy as np
from scipy.spatial.distance import squareform

import torch

import tqdm
import pandas as pd
from methods.active_learning import uniform_random, active_represent
from utils.io_utils import load_model, save_model, find_latest_checkpoint
from itertools import combinations


models = {
          'uniform_random': uniform_random.UniformRandom,
          "active_representation": active_represent.Active_Representation
          }

acquisition_fns = ['thompson', 'variance', 'optimism', 'pessimism','random', "hybrid",
                   "optimism-hall", "optimism-div", "eps-greedy", "greedy-div","greedy-elim-div",
                   "greedy-explore", "bait", "variance_mixed", "badge"]

def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--interaction_file", type=str, required=True,
                        help="path to file with interaction vector")
    parser.add_argument("-nr", "--n_rounds", type=int, default=10,
                        help="how many rounds of active learning to perform")
    parser.add_argument("-ns", "--n_samples", type=int, default=200,
                        help="number of new observations to acquire in each round of active learning")
    parser.add_argument("--model", type=str, required=True, choices=models.keys(),
                        help="which model to use")
    parser.add_argument("--acquisition", type=str, default='thompson', choices=acquisition_fns,
                        help="which acquisition function to use")
    parser.add_argument("--config_file", type=str,
                        help="configuration file for the model")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help="where to save output")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="random seed")
    parser.add_argument('--gpu', type=int, required=True, help='GPU id to use')
    return parser


def main(args):    

    # load config 
    with open(args.config_file) as f:
        config = json.load(f)

    # Set the device
    # device = 'cpu'
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    n_genes = 50
    n_pairs = int((n_genes * (n_genes - 1)) / 2)
    n_samples, n_rounds = args.n_samples, args.n_rounds
    outfile = args.outfile
    outfile_dir = os.path.dirname(outfile)
    infile = args.interaction_file
    acquisition_fn = args.acquisition


    # observe interactions from one bootstrap run
    if os.path.exists(infile):    
        # loaded in the interaction vector
        print("loading the interaction vector file",flush=True)
        interaction_vector = np.load(infile)
        interaction_vector = torch.from_numpy(interaction_vector).float()

    # 1 = we have seen and will train on this sample, 
    # 0 = hidden
    # initially, everything is hidden
    indices = np.arange(n_pairs)
    seen_mask = np.zeros(n_pairs, dtype=bool)


    # TODO:
    # To modify the code so all methods can be run in the same framework
    if args.model == "active_representation":
        embedding_dir = os.path.join(config['base_embedding_dir'],
                                    args.config_file.split("/")[-1].split(".")[0],
                                    acquisition_fn, "{}".format(args.seed))
        os.makedirs(embedding_dir, exist_ok=True)

        # load model
        checkpoint_dir = os.path.join(config['base_checkpoint_dir'],
                                    args.config_file.split("/")[-1].split(".")[0],
                                    f"batch_size_{n_samples}_n_rounds_{n_rounds}",
                                    acquisition_fn, 
                                    "div_False",
                                    # "div_{}".format(args.diversification),
                                    "r{}".format(args.seed))
        
        pred_dir = os.path.join(config['base_pred_dir'],
                                args.config_file.split("/")[-1].split(".")[0],
                                f"batch_size_{n_samples}_n_rounds_{n_rounds}",
                                acquisition_fn, 
                                "div_False",
                                # "div_{}".format(args.diversification),
                                "r{}".format(args.seed)
                                )
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)


    model = models[args.model](model_config=args.config_file, device=device)

    # resume training from last active learning round, if applicable
    if os.path.exists(outfile):
        if args.model == "active_representation":
            # loadest the latest model 
            latest_step, latest_file = find_latest_checkpoint(directory_path=checkpoint_dir)
            model = load_model(checkpoint_dir, latest_file, model)
            if model.representation_model_config['model_class'] == "ActiveFE":
                with torch.no_grad():
                    model.nodes_embedding = model.model.representation_model().detach().clone()
            else:
                with torch.no_grad():
                    model.nodes_embedding = model.model.representation_model.encode(model.model.data.x, 
                                                                                model.model.data.edge_index, 
                                                                                model.model.data.edge_type).detach().clone()
                    if model.hiv_indices is not None:
                        model.nodes_embedding = model.nodes_embedding[model.hiv_indices]

        # load in the seen samples and interaction vectors that are consistent with latestest model
        with open(outfile, 'r+') as f:
            pos = f.tell()  
            for last_round in range(latest_step + 2):
                line = f.readline()  
                new_samples = np.array([int(sample) for sample in line.strip().split(",")[1:]])
                seen_mask[new_samples] = True
                pos = f.tell()  
            f.seek(pos)  
            f.truncate()  # truncate from this position onward to ensure 
                          # outfile consistent with the latest model

    else:
        # initializat the model 
        if (args.model == "active_representation") and ("noinitialtraining" not in outfile):
            print("initialize the model through negative sampling", flush=True)
            model.init_model()
            save_model(checkpoint_dir, 'model_init.pt', model)

        last_round = 0
        # random initialization
        random_file_name = f"random-final_batchsize_{n_samples}_numrounds_{n_rounds}_r{args.seed}.csv"
        random_file = os.path.join(outfile_dir, random_file_name)
        if os.path.exists(random_file):
            print(f"loading the initial random samples for replicate {args.seed}")
            # Read the first row separately
            with open(random_file, 'r') as file:
                first_row = file.readline().strip().split(',')[1:]
            first_row_integers = [int(value) for value in first_row]
            new_samples = np.array(first_row_integers)
        else:
            new_samples = np.random.choice(indices, size=n_samples, replace=False)
        seen_mask[new_samples] = True
        with open(outfile, 'w') as f:
            f.write('0,' + ','.join([str(ix) for ix in new_samples]) + '\n')

    logging_interval = 1
    pbar = tqdm.tqdm(range(last_round, n_rounds), position=0, leave=True)
    indices_all = np.arange(len(seen_mask))
    for i in pbar:
        pbar.set_description('Round %s' % (i + 1))
        # always fit the model from scratch in the first iteration
        if args.model == "active_representation":
            if i == 0: 
                model.fit(interaction_vector, seen_mask)

            else:
                model.update(interaction_vector, seen_mask, new_samples)
            indices = np.arange(len(seen_mask))[~seen_mask]
            predictions = model.predict(copy.deepcopy(indices)).cpu().detach().numpy()
            predictions_all = model.predict(copy.deepcopy(indices_all)).cpu().detach().numpy()
            np.save(os.path.join(pred_dir,f"predictions_step_{i}.npy"), predictions)
            np.save(os.path.join(pred_dir,f"predictions_all_step_{i}.npy"), predictions_all)
            if i<n_rounds-1:
                new_samples = model.acquisition(n_samples, seen_mask, n_round=i/(n_rounds-1), method=acquisition_fn)
            else:            
                # in the last round perform greedy prediction
                model.representation_model_config["params"]["eps"] = 0
                # need to modify the n_round to be consistent 
                new_samples = model.acquisition(n_samples, seen_mask, n_round=1, method='eps-greedy')
            save_model(checkpoint_dir, 'model_step_{}.pt'.format(i), model)


            # decay the learning rate 
            # model.training_config["learning_rate_regression"] = max(model.training_config["learning_rate_regression"]*model.training_config["learning_rate_decay"],1e-3)
            # model.training_config["learning_rate_embedding"] = max(model.training_config["learning_rate_embedding"]*model.training_config["learning_rate_decay"],1e-5)
        else:
            new_samples = model.acquisition(n_samples, seen_mask, method=acquisition_fn)
        # we should NOT have seen these samples before
        assert seen_mask[new_samples].sum() == 0
        # there should not be any duplicate samples
        assert len(set(new_samples)) == n_samples
        # update mask
        seen_mask[new_samples] = True
        with open(outfile, 'a') as f:
            f.write(str(i + 1) + ',' + ','.join([str(ix) for ix in new_samples]) + '\n')
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
