import pickle 
import json 

import os.path as osp

from tqdm import tqdm

import numpy as np 

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GAE, RGCNConv, SAGEConv, to_hetero, GraphNorm, DiffGroupNorm
# from torch_geometric.nn.conv import CuGraphRGCNConv

from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree

from sklearn.model_selection import train_test_split


""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""


# TODOs:
#      1. Test the newly added prediction layer 
#      2. mod the prediction layer to general bilinear form instead of assuming
#         a square matrix 

class RGCNEncoder(torch.nn.Module):
    # TODO:
    #     1. add feature to allow multiple layers of hidden dimensions 
    def __init__(self, num_nodes, node_dim, emb_dim, num_relations, hidden_dim=128, num_conv_layers=1, device="cpu"):
        '''
            num_nodes: number of nodes
            node_dim: dimension of the representation of the nodes 
            emb_dim:  dimension of final embedding of the nodes 
            num_relations: number of relations of the graph 
            hidden_dim: dimensions of hidden layers
            num_conv_layers: number of convolcution layer 
        '''
        super().__init__()
        self.device = device 
        
        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(RGCNConv(num_nodes, hidden_dim, num_relations))
        # self.conv_layers.append(CuGraphRGCNConv(num_nodes, hidden_dim, num_relations)) # more efficient library
        # self.norm_layers.append(GraphNorm(hidden_dim))
        self.norm_layers.append(DiffGroupNorm(in_channels=hidden_dim,groups=3))
        for _ in range(num_conv_layers - 2):
            self.conv_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            # self.conv_layers.append(CuGraphRGCNConv(hidden_dim, hidden_dim, num_relations)) # more efficient library
            # self.norm_layers.append(GraphNorm(hidden_dim))
            self.norm_layers.append(DiffGroupNorm(in_channels=hidden_dim,groups=3))
        self.conv_layers.append(RGCNConv(hidden_dim, emb_dim, num_relations))
        # self.conv_layers.append(CuGraphRGCNConv(hidden_dim, emb_dim, num_relations))
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.node_emb)
        for layer in self.conv_layers:
            layer.reset_parameters()
        for layer in self.norm_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        # x= self.node_emb[node_idx,:]
        # memory_model_init = torch.cuda.memory_allocated(self.device)
        # for conv in self.conv_layers[:-1]:
        for conv,norm in zip(self.conv_layers[:-1],self.norm_layers):
            # x = conv(x, edge_index, edge_type).relu()
            x = norm(conv(x, edge_index, edge_type)).relu()
            # memory_model_init = torch.cuda.memory_allocated(self.device)
        x = self.conv_layers[-1](x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, input_dim, device="cpu"):
        ''' 
            num_relations: number of relations
            hidden_channels: number of hidden channels
        '''
        super().__init__()
        self.device = device
        
        self.rel_emb = Parameter(torch.empty(num_relations, input_dim, device=self.device)) # DistMult parameters
        self.reset_parameters()
        self.num_relations = num_relations

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type, sigmoid=False):
        ''''
            z:         the latent representation of the nodes
            edge_idx:  the pair of edge index we are to compute 
                        the score 
            edge_type: the type of relation 
        '''
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]        
        rel = self.rel_emb[edge_type]
        return torch.sigmoid(torch.sum(z_src * rel * z_dst, dim=1)) if sigmoid else torch.sum(z_src * rel * z_dst, dim=1)


def negative_sampling(edge_index, num_nodes):
    """Sample edges by corrupting either the subject or the object of each edge.
        edge_index: the set of positive edges
        num_nodes:  number of nodes 
    """
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1
    
    neg_edge_index = edge_index.clone()
    
    neg_0_mask = torch.randint(num_nodes, (mask_1.sum(), ),device=neg_edge_index.device)
    index_same = (neg_0_mask == edge_index[0, mask_1])
    while index_same.sum() != 0:
        neg_0_mask[index_same] = torch.randint(num_nodes, (index_same.sum(), ), device=neg_edge_index.device)
        index_same = (neg_0_mask == edge_index[0, mask_1])

    neg_1_mask = torch.randint(num_nodes, (mask_2.sum(), ), device=neg_edge_index.device)
    index_same = (neg_1_mask == edge_index[1, mask_2])
    while (neg_1_mask == edge_index[1, mask_2]).sum() != 0:
        neg_1_mask[index_same] = torch.randint(num_nodes, (index_same.sum(), ),device=neg_edge_index.device)
        index_same = (neg_1_mask == edge_index[1, mask_2])

    neg_edge_index[0, mask_1] = neg_0_mask
    neg_edge_index[1, mask_2] = neg_1_mask

    return neg_edge_index


def train():
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.edge_index, data.edge_type)
    pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

    neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
    neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    embeddings = model.encode(data.edge_index, data.edge_type)
    test_mrr = compute_mrr(embeddings, data.test_edge_index, data.test_edge_type)

    return test_mrr, embeddings


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad() 
def compute_mrr(z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False
        
        tail_mask[src] = False
        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False
        
        head_mask[dst] = False
        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()


if __name__ == '__main__':
    is_hiv_data = True 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("device:",device)

    if is_hiv_data:
        # with open("/usr/workspace/PROTECT/hetgnn/data/HIV_hetergraph_gene_bp.dt","rb") as f:
        with open("/usr/workspace/PROTECT/hetgnn/data/HIV_hetgraph_BP_Gene_Protein_RandWalk_5by5_20240909.dt","rb") as f:
            data_full = pickle.load(f)
        data = data_full[0]
        hiv_indices = data_full[1]
        # with open("/usr/workspace/zhu18/protect_si_recommender/HGNN/data/HIV/HIV_hetergraph.dt","rb") as f:
        # # with open("../data/HIV/HIV_hetergraph.dt","rb") as f:
        #     data = pickle.load(f)
        data = data.to_homogeneous()
        data.x = torch.eye(data.num_nodes)
        alledge = list(range(len(data.edge_type)))
        train_i, test_i = train_test_split(alledge, test_size=0.2, shuffle=True)

        train_edge_index=data.edge_index[:,train_i].tolist()
        train_edge_type=data.edge_type[train_i].tolist()

        test_edge_index=data.edge_index[:,test_i].tolist()
        test_edge_type=data.edge_type[test_i].tolist()

        data.train_edge_index = torch.tensor(train_edge_index)
        data.train_edge_type = torch.tensor(train_edge_type)
        data.test_edge_index =torch.tensor(test_edge_index)
        data.test_edge_type = torch.tensor(test_edge_type)
        data.to(device)

    else:
        from torch_geometric.datasets import RelLinkPredDataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'RLPD')
        dataset = RelLinkPredDataset(path, 'FB15k-237')
        data = dataset[0].to(device)
        print("\n data: \n",data)

    loader = NeighborLoader(
            data,
            num_neighbors=[30]*2,
            batch_size=10,
            replace=False,
            shuffle=False,
        )

    embedding_d = 50 # dimension of the embedding 
    node_dim =  200 # initial dimension to represent the node 

    model_config = "/usr/workspace/zhu18/protect_si_recommender/data/active_representation_learning/active_represent_bilinear_BP_Gene_Protein_raw_data_nopruning_test.json"
    with open(model_config) as f:
            model_config = json.load(f)

    # config for representation and regression models
    representation_params = model_config['model']["representation_model"]['params']

    embedding_d = representation_params["embedding_d"] # dimension of the embedding 
    node_dim =  representation_params["node_dim"] # initial dimension to represent the node 
    num_relations = len(set(data.edge_type.tolist()))

    model = GAE(
        RGCNEncoder(data.num_nodes, node_dim, embedding_d, num_relations, hidden_dim=representation_params["hidden_dim"], num_conv_layers=representation_params["num_conv_layers"]),
        DistMultDecoder(len(set(data.edge_type.tolist())), embedding_d),
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    print("starting to train the inital embedding................................")
    for epoch in range(300):
        optimizer.zero_grad()
        # z = self.model.encode(self.data.edge_index, self.data.edge_type)
        z = model.encode(data.x, data.edge_index, data.edge_type)
        pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

        neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
        neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        pen_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean() # penalty loss
        loss = cross_entropy_loss + 1e-2 * pen_loss

        loss.backward()
        
        optimizer.step()
        # scheduler.step(loss.item())
        if epoch%20==1:
            print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_type)
        pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)
        neg_out = model.decode(z, neg_edge_index, data.train_edge_type)
        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        print(F.binary_cross_entropy_with_logits(out, gt))

    neg_test_edge_index = negative_sampling(data.test_edge_index, data.num_nodes)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_type)
        pos_out = model.decode(z, data.test_edge_index, data.test_edge_type)
        neg_out = model.decode(z, neg_test_edge_index, data.test_edge_type)
        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        print(F.binary_cross_entropy_with_logits(out, gt))
