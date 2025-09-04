import os
import pickle

import numpy as np
from numpy.random import default_rng

import torch
from torch_geometric.data import HeteroData

def generate_data(num_genes=50, num_replicates=10, seed=1234):
    """ function to generate the synthetic data 

    inputs: 
    num_genes: number of genes of interests
    num_replicates: number of replicates
    """


    ## genearting the interaction vectors 
    rng = default_rng(seed)
    for replicate in range(num_replicates):
        interaction_matrix = 10*np.abs(rng.random((num_genes,num_genes)))
        interaction_matrix = interaction_matrix.T+interaction_matrix
        row_idx, col_idx = np.triu_indices(num_genes, k=1)

        # Extract elements above the diagonal as a vector
        interaction_vector = interaction_matrix[row_idx, col_idx]
        np.save(os.path.join("data","interaction_vector_r{}.npy".format(replicate)), interaction_vector)
    
    ## generating the knwoledge graph

    # Create an empty HeteroData object
    data = HeteroData()

    data['genes'].num_nodes = num_genes

    # Add edges between user and item (e.g., "rates" relation)
    num_edges = 50
    src = torch.randint(0, num_genes, (num_edges,))
    dst = torch.randint(0, num_genes, (num_edges,))
    data['genes', 'upregulate', 'genes'].edge_index = torch.stack([src, dst], dim=0)

    # Optionally, add another edge type (e.g., "follows" between users)
    num_down = 50
    src_down = torch.randint(0, num_genes, (num_down,))
    dst_down = torch.randint(0, num_genes, (num_down,))
    data['genes', 'downregulate', 'genes'].edge_index = torch.stack([src_down, dst_down], dim=0)

    with open('data/kg.pt', "wb") as f:
        pickle.dump(data, f)
    return 

if __name__ == '__main__':

  generate_data()