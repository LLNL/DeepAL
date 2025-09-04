import math 

import torch
import torch.nn as nn
from torch.nn import Parameter

import torch.nn.functional as F
import torch.nn.init as init


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=1):
        super(SiameseNetwork, self).__init__()
        
        # Define the branches of the Siamese network
        self.branch1 = self.create_branch(input_dim, hidden_dim)
        self.branch2 = self.create_branch(input_dim, hidden_dim)
        self.distance = nn.PairwiseDistance(p=2)

        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),  # Assuming concatenation
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output a single value
        )

    def create_branch(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x1, x2):
        # Forward pass through both branches
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        # Compute Euclidean distance
        distance_output = self.distance(x1, x2).view(-1, 1)

        combined = torch.cat((out1, out2, distance_output), dim=1)  # Concatenate branch outputs

        output = self.regressor(combined)
        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                       (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class InnerProductLayer(nn.Module):
    def __init__(self, input_dim):
        super(InnerProductLayer, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
        
    def forward(self, x1, x2):
        # inner product layer
        if x1.dim() == 1:
            # Perform dot product for 1D vectors
            # output = torch.sum(self.dropout(x1 * x2))+self.bias
            output = torch.sum(x1 * x2)+self.bias
        else:
            # output = torch.sum(self.dropout(x1 * x2), dim=1, keepdim=True)+self.bias
            output = torch.sum(x1 * x2, dim=1, keepdim=True)+self.bias
        return output
    
    def reset_parameters(self):
        # Zero initialization for biase
        init.uniform_(self.bias, 0, 0)

class DiagonalBilinearLayer(nn.Module):
    def __init__(self, input_dim):
        super(DiagonalBilinearLayer, self).__init__()
        # Initialize the diagonal weights and bias with the same shape as input dimension
        self.diag_weights = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        # self.dropout = nn.Dropout(0.1)
        
        # Initialize weights and bias
        self.reset_parameters()
        
    def reset_parameters(self):

        # Kaiming uniform initialization for weights
        init.kaiming_uniform_(self.diag_weights.unsqueeze(0).unsqueeze(0), a=math.sqrt(5))
        # Zero initialization for biases
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.diag_weights.unsqueeze(0).unsqueeze(0))
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x1, x2):
        # Perform element-wise multiplication of inputs
        elementwise_mul = x1 * x2
        # Scale by the diagonal weights and sum
        # output = torch.sum(self.dropout(elementwise_mul * self.diag_weights), dim=1, keepdim=True) + self.bias
        output = torch.sum(elementwise_mul * self.diag_weights, dim=1, keepdim=True) + self.bias
        return output

class BiLinearRegressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1, sym=True):
        super(BiLinearRegressionNetwork, self).__init__()
        self.sym = sym

        # Define the Bilinear Regression layers with dropout
        self.bilinear= nn.Bilinear(input_dim, input_dim, output_dim)
        # self.dropout = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.1)
        # self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        # self.dropout = nn.Dropout(0.2)


    def forward(self, x,y):
        x = self.dropout(x)
        y = self.dropout(y)
        if self.sym:
            return F.softplus(self.bilinear(x,y)+self.bilinear(y,x))+self.bias
            # return (self.bilinear(x,y)+self.bilinear(y,x)).relu()+self.bias
            # return (self.bilinear(x,y)+self.bilinear(y,x)).relu()*self.scale+self.bias
            # return self.dropout(self.bilinear(x,y))+self.dropout(self.bilinear(y,x))
            # return self.dropout(self.bilinear(x,y))+self.dropout(self.bilinear(y,x))
        else: 
            return self.bilinear(x,y).relu()*self.scale+self.bias
            # return self.dropout(self.bilinear(x,y))
        
    def reset_parameters(self):
        # reset the parameters
        self.bilinear.reset_parameters()
        return 

class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(RegressionNetwork, self).__init__()

        # Define the regression layers
        super().__init__()
        # self.regressor = nn.Sequential(
        #     nn.Linear(encoded_dim, hidden_dim),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.Dropout(0.5),
        # )
        self.regressor1 = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.Dropout(0.5)
                )
        self.regressor2 = nn.Sequential(
                            nn.Linear(input_dim, output_dim),
                            nn.Dropout(0.5)
                        )

    def forward(self, x, y):
        return self.regressor1(x)+self.regressor2(y)
    

class BayesBiLinearRegressionNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1, sym=True, device="cpu"):
        ''' 
            num_relations: number of relations
            hidden_channels: number of hidden channels
        '''
        super().__init__()
        self.device = device
        self.sym = sym
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the bilinear parameters probabilistically with a prior distribution
        self.weight_vector_dist = dist.Normal(torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device)).expand([input_dim, input_dim])
        self.bias_dist = dist.Normal(torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
        self.bias = None 
        self.weights = None 

    def set_param(self, weights):
        self.weights = weights

    def forward(self, x, y, resample=True):
        if resample or self.weights == None:    
            with pyro.plate("output_dim", self.output_dim, dim=-1):
                self.weights = pyro.sample(f"bilinear_w", self.weight_vector_dist.to_event(2))
                self.bias = pyro.sample(f"bilinear_bias", self.bias_dist.to_event(1))

        if self.sym:
            return F.softplus(F.bilinear(x,y,self.weights)+self.bilinear(y,x,self.weights))+self.bias
        else:
            return F.softplus(F.bilinear(x,y, self.weights))+self.bias