import pickle
import os
import io

del os.environ['OMP_PLACES']
del os.environ['OMP_PROC_BIND']
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

import socket 

import copy 

import argparse
import json

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import trim_mean 
from scipy.stats.mstats import trimmed_std

import torch 
import torch.nn as nn 
import torch.distributed as dist

import time
import datetime

import tqdm
import pandas as pd

from methods import active_learning
from methods.active_learning import uniform_random, active_represent
from utils.io_utils import load_model, save_model, find_latest_checkpoint

from itertools import combinations

def custom_parser(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Determine the maximum number of columns
    max_columns = max(len(line.strip().split(',')) for line in lines)
    adjusted_lines = []
    
    for line in lines:
        columns = line.strip().split(',')
        # Add empty strings to match the max number of columns
        columns += [''] * (max_columns - len(columns))
        adjusted_lines.append(','.join(columns))
    
    # Join adjusted lines and read into DataFrame
    adjusted_content = '\n'.join(adjusted_lines)
    df = pd.read_csv(io.StringIO(adjusted_content), header=None, index_col=0)
    
    # Convert all columns to nullable integer type
    df = df.apply(pd.to_numeric, errors='coerce').convert_dtypes()
    
    return df

def mean_ignore_min_max_2d(arr):
    if arr.shape[0] <= 2:
        raise ValueError("Each column must have more than two elements to ignore min and max values.")
    
    # Sort along axis=0 (each column independently)
    sorted_arr = np.sort(arr, axis=0)
    
    # Exclude the first (min) and last (max) elements in each column
    trimmed_arr = sorted_arr[1:-1]
    
    # Calculate the mean of the trimmed array along axis=0
    return np.mean(trimmed_arr, axis=0)

def rmse(y_pred, y_target):
    return np.sqrt(np.mean((y_pred-y_target)**2))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


model_dict = {
          'uniform_random': uniform_random.UniformRandom,
          "active_representation": active_represent.Active_Representation
          }

acquisition_fns = ['thompson', 'variance', 'optimism', 'pessimism','random', "hybrid",
                   "optimism-hall", "optimism-div", "eps-greedy", "greedy-div","greedy-elim-div",
                   "greedy-explore", "bait", "variance_mixed", "annealing", "max-optimism", "badge"]

def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--interaction_file", type=str, required=True,
                        help="path to file with interaction vector")
    parser.add_argument("-nr", "--n_rounds", type=int, default=10,
                        help="how many rounds of active learning to perform")
    parser.add_argument("-ns", "--n_samples", type=int, default=200,
                        help="number of new observations to acquire in each round of active learning")
    parser.add_argument("--model", type=str, required=True, choices=model_dict.keys(),
                        help="which model to use")
    parser.add_argument("--acquisition", type=str, default='thompson', choices=acquisition_fns,
                        help="which acquisition function to use")
    parser.add_argument("--config_file", type=str,
                        help="configuration file for the model")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help="where to save output")
    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="random seed")
    return parser

# Set up distributed training environment
def setup():
    timeout = datetime.timedelta(minutes=45)
    dist.init_process_group(backend="gloo", init_method="env://",
                            world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
                            rank=int(os.environ['OMPI_COMM_WORLD_RANK']),
                            timeout=timeout)
    # dist.init_process_group(backend="nccl", init_method="env://",
    #                         world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
    #                         rank=int(os.environ['OMPI_COMM_WORLD_RANK']))
    rank = dist.get_rank()

    # We no longer need to manually set local rank or assign devices
    # The visible GPU is handled by lrun via CUDA_VISIBLE_DEVICES
    device = torch.device('cuda')  # Use the only visible CUDA device

    # Print debug info
    # print(f"Process {rank} is set up and using {device} on {socket.gethostname()}, "
    #       f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}", flush=True)
    
    return device

# Broadcast data on CPU to all processes
def broadcast_data(rank, data):
    """
    Broadcasts tensors from rank 0 to all other ranks on the CPU.
    Each process will later move the data to GPU for training.
    """
    for tensor in data:
        dist.broadcast(tensor, src=0)  

# use to decide the model to be loaded in 
def gather_and_broadcast_min(world_size, local_scalar):
    # Step 1: Gather all scalars (in scalar form)
    gathered_scalars = [0 for _ in range(world_size)]
    dist.all_gather_object(gathered_scalars, local_scalar.item())
    dist.barrier()

    # Step 2: Find the minimum value on the root process (rank 0)
    rank = dist.get_rank()
    if rank == 0:
        min_scalar = min(gathered_scalars)
    else:
        min_scalar = 0  # Dummy value to be replaced

    # Step 3: Broadcast the minimum value to all processes
    min_scalar = torch.tensor(min_scalar, dtype=torch.int32)
    dist.barrier()
    dist.broadcast(min_scalar, src=0)

    # Convert back to scalar for easy use
    return min_scalar.item()

# Cleanup after training
def cleanup():
    dist.destroy_process_group()

def acquisition(i, y_opt, y_std, y_mean, y_min, n_samples, n_rounds, indices, gathered_norms=None, acquisition_fn="eps-greedy"):
    if acquisition_fn == "optimism":
        new_samples = indices[np.argsort(y_opt)][:n_samples]
    elif acquisition_fn == "variance":
        new_samples = indices[np.argsort(y_std)][-n_samples:]
    elif acquisition_fn == "badge":
        grad_norms =  np.vstack([gathered_norm.numpy() for gathered_norm in gathered_norms])
        # grad_norms_mean = np.mean(grad_norms, axis=0)
        grad_norms_mean = np.median(grad_norms, axis=0)
        new_samples = indices[np.argsort(grad_norms_mean)][-n_samples:]
    elif acquisition_fn == "greedy-explore":
        grad_norms =  np.vstack([gathered_norm.numpy() for gathered_norm in gathered_norms])
        # grad_norms_mean = np.mean(grad_norms, axis=0)
        grad_norms_mean = np.median(grad_norms, axis=0)
        if i<n_rounds-1:
            num_greedy = int(n_samples*i/(n_rounds-1))
            num_explore = n_samples-num_greedy
            new_samples_greedy = indices[np.argsort(y_mean)][:num_greedy]
            indices_explore = np.setdiff1d(indices,new_samples_greedy,assume_unique=True)
            y_mean = y_mean[np.array([np.where(indice==indices)[0][0] for indice in indices_explore])]
            grad_norms_mean = grad_norms_mean[np.array([np.where(indice==indices)[0][0] for indice in indices_explore])]
            new_samples_explore = indices_explore[np.argsort(grad_norms_mean)][-num_explore:]
            new_samples = np.concatenate([new_samples_greedy, new_samples_explore])
        else: 
            new_samples = indices[np.argsort(y_mean)][:n_samples]
    elif acquisition_fn == "eps-greedy":
        new_samples = indices[np.argsort(y_mean)][:n_samples]
    elif acquisition_fn == "max-optimism":
        new_samples = indices[np.argsort(y_min)][:n_samples]
    elif acquisition_fn == "annealing":
        if i<n_rounds-1:
            new_samples = indices[np.argsort(y_std)][-n_samples:]
        else: 
            new_samples = indices[np.argsort(y_min)][:n_samples]
    return new_samples 

def main(args):    
    
    device = setup()
    world_size = dist.get_world_size()
    rank = dist.get_rank()    
    # load config 
    with open(args.config_file) as f:
        config = json.load(f)

    n_genes = 50
    n_pairs = int((n_genes * (n_genes - 1)) / 2)
    n_samples, n_rounds = args.n_samples, args.n_rounds
    outfile = args.outfile
    outfile_dir = os.path.dirname(outfile)
    infile = args.interaction_file
    acquisition_fn = args.acquisition

    # 1 = we have seen and will train on this sample, 
    # 0 = hidden
    # initially, everything is hidden
    indices = np.arange(n_pairs)
    seen_mask = torch.zeros(n_pairs, dtype=bool)

    # Only allow the master rank the process the data to avoid race condition
    if rank == 0:
        pred_dir = os.path.join(config['base_pred_dir'],
                                args.config_file.split("/")[-1].split(".")[0],
                                f"batch_size_{n_samples}_n_rounds_{n_rounds}",
                                acquisition_fn, 
                                "r{}".format(args.seed)
                                )
        os.makedirs(pred_dir, exist_ok=True)
        # observe interactions from one bootstrap run
            # observe interactions from one bootstrap run
        if os.path.exists(infile):    
            # loaded in the interaction vector
            print("loading the interaction vector file",flush=True)
            interaction_vector = np.load(infile)
            interaction_vector = torch.from_numpy(interaction_vector).float()


    dist.barrier() # put a barrier to ensure data pre-processing step is completed
    if rank != 0:
        interaction_vector = np.load(infile)
        interaction_vector = torch.from_numpy(interaction_vector).float()

    model = model_dict[args.model](model_config=args.config_file, device=device)
    checkpoint_dir = os.path.join(config['base_checkpoint_dir'],
                                        args.config_file.split("/")[-1].split(".")[0],
                                        f"batch_size_{n_samples}_n_rounds_{n_rounds}",
                                        acquisition_fn, 
                                        "{}_process_{}".format(args.seed, rank))
    # resume training from last active learning round, if applicable
    if_outfile_exists = os.path.exists(outfile)
    if_initial = ("noinitialtraining" not in outfile) and (model.representation_model_config['model_class'] == "ActiveGNN")
    # print("the model pass in without problem for gpu: {}".format(rank))
    os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier() # prevent racing for checking if the outfile exists 
    if if_outfile_exists:

        df = custom_parser(outfile)
        num_completed_round = len(df)-2

        # print(f"output file exists, trying to load the model for model: {rank}")
        # loadest the latest model 
        latest_step, latest_file = find_latest_checkpoint(directory_path=checkpoint_dir)
        latest_step_tensor = torch.tensor(latest_step, dtype=torch.int32) 
        dist.barrier()
        min_value = gather_and_broadcast_min(world_size, latest_step_tensor)
        dist.barrier()
        min_value = min(min_value, num_completed_round)
        if min_value == -1: 
            latest_file = "model_init.pt"
        else: 
            latest_file = "model_step_{}.pt".format(min_value)
        latest_step = min_value 
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
        if rank==0:
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
            last_round = latest_step if latest_step == 0 else latest_step+1
    else:
        # initializat the model 
        if if_initial: 
            # print(f"initialize the model through negative sampling for {rank}", flush=True)
            model.init_model()
            save_model(checkpoint_dir, 'model_init.pt', model)
        # random initialization
        last_round = 0
        if rank==0:

            # ensuring same replicate number across methods have the same initial
            # random samples 
            random_file_name = f"random-final_batchsize_{n_samples}_numrounds_{n_rounds}_r{args.seed}.csv"
            random_file = os.path.join(outfile_dir, random_file_name)
            if os.path.exists(random_file):
                # print(f"loading the initial random samples for replicate {args.seed}")
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
    indices_all = np.arange(len(seen_mask))

    # Synchronize all processes to before data is broadcasted
    dist.barrier()
    # broadcast the seen_masks 
    broadcast_data(rank, [seen_mask])
    # Synchronize all processes to ensure data is broadcasted before proceeding
    dist.barrier()
    # print(f"for rank {rank} seen mask shape is {seen_mask.shape} with total seen samples {torch.sum(seen_mask)}")
    # logging_interval = 1
    pbar = tqdm.tqdm(range(last_round, n_rounds), position=0, leave=True) if rank==0 else tqdm.tqdm(range(last_round, n_rounds), position=0, leave=True, disable=True)
    for i in pbar:
        pbar.set_description('Round %s' % (i + 1))
        # always fit the model from scratch in the first iteration
        # or for models that don't have an update method

        if args.model == "active_representation":
            # find indices of points that have not been seen yet
            indices = np.arange(len(seen_mask))[(~seen_mask)]
            model.fit(interaction_vector, copy.deepcopy(seen_mask), ensemble_size=world_size)
            save_model(checkpoint_dir, 'model_step_{}.pt'.format(i), model)
            if i == (n_rounds-1):
                embedding_dir = os.path.join(config['base_embedding_dir'],
                                args.config_file.split("/")[-1].split(".")[0],
                                f"batch_size_{n_samples}_n_rounds_{n_rounds}",
                                acquisition_fn, 
                                "{}_process_{}".format(args.seed, rank))
                os.makedirs(embedding_dir, exist_ok=True)
                embedding_file = 'final_embeddings.npy'
                # torch.save(model.nodes_embedding.cpu(), os.path.join(embedding_dir, embedding_file))
                np.save(os.path.join(embedding_dir, embedding_file), model.nodes_embedding.cpu().numpy())
            if acquisition_fn in ["greedy-explore", "badge"]:
                predictions, norms = model.predict(copy.deepcopy(indices), return_grad_norm=True)
                predictions = predictions.cpu().detach().numpy()
                norms = norms.cpu().detach().numpy()
            else:
                predictions = model.predict(copy.deepcopy(indices)).cpu().detach().numpy()
            predictions_all = model.predict(copy.deepcopy(indices_all)).cpu().detach().numpy()
        else:
            new_samples = model.acquisition(n_samples, seen_mask, method=acquisition_fn)
         # Barrier to ensure all processes have completed prediction before gathering
        dist.barrier()

        # Gather predictions from all processes to rank 0 (master node)
        gathered_predictions = [torch.zeros_like(torch.from_numpy(predictions)) for _ in range(world_size)]
        dist.gather(torch.from_numpy(predictions), gather_list=gathered_predictions if rank == 0 else None, dst=0)

        if acquisition_fn in ["greedy-explore", "badge"]:
            dist.barrier()
            gathered_norms = [torch.zeros_like(torch.from_numpy(norms)) for _ in range(world_size)]
            dist.gather(torch.from_numpy(norms), gather_list=gathered_norms if rank == 0 else None, dst=0)

        # Barrier to ensure all processes finish gathering before moving on
        dist.barrier()    

        gathered_predictions_all = [torch.zeros_like(torch.from_numpy(predictions_all)) for _ in range(world_size)]
        dist.gather(torch.from_numpy(predictions_all), gather_list=gathered_predictions_all if rank == 0 else None, dst=0)

        # Barrier to ensure all processes finish gathering before moving on
        dist.barrier()   
        if rank == 0:
            # y_ground_truth = interaction_vector_np[np.arange(len(seen_mask))[~seen_mask]]

            y_preds =  np.vstack([gathered_prediction.numpy() for gathered_prediction in gathered_predictions])
            y_preds_all =  np.vstack([gathered_prediction_all.numpy() for gathered_prediction_all in gathered_predictions_all])
            np.save(os.path.join(pred_dir,f"predictions_step_{i}.npy"), y_preds)
            np.save(os.path.join(pred_dir,f"predictions_all_step_{i}.npy"), y_preds_all)

            y_min = np.quantile(y_preds, q=0.1, axis=0)
            y_mean = trim_mean(y_preds, proportiontocut=0.1, axis=0)
            y_std = trimmed_std(y_preds, axis=0)
            y_std = np.std(y_preds, axis=0)
            y_opt = y_mean - y_std
            new_samples = acquisition(i, y_opt, y_std, y_mean, y_min, n_samples, n_rounds, copy.deepcopy(indices), acquisition_fn=acquisition_fn)

            # we should NOT have seen these samples before
            assert seen_mask[new_samples].sum() == 0
            # there should not be any duplicate samples
            assert len(set(new_samples)) == n_samples
            # update mask
            seen_mask[new_samples] = True
            with open(outfile, 'a') as f:
                f.write(str(i + 1) + ',' + ','.join([str(ix) for ix in new_samples]) + '\n')

        dist.barrier()    
        broadcast_data(rank, [seen_mask])
        dist.barrier()   
        # print(f"for rank {rank} seen mask shape is {seen_mask.shape} with total seen samples {torch.sum(seen_mask)}", flush=True)
    cleanup()
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    try:
        main(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
