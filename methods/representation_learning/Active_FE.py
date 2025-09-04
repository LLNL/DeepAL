import math 

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from methods.representation_learning.FE import FE 
from methods.representation_learning.regression import DiagonalBilinearLayer, InnerProductLayer, BiLinearRegressionNetwork, RegressionNetwork

from copy import deepcopy



regression_classes = {"Linear": RegressionNetwork,
                      "BiLinear_diagonal": DiagonalBilinearLayer, 
                      "Inner_product": InnerProductLayer, 
                      "Bilinear": BiLinearRegressionNetwork
                    }
class ActiveFE(torch.nn.Module):
    def __init__(self, num_nodes=356, embedding_d=50, update_embedding=True,
                 reg_class="Bilinear", device="cpu"):
        """

        """
        super().__init__()
        self.device = device 

        self.update_embedding = update_embedding
        self.reg_class = reg_class
        self.input_dim = embedding_d

        self.representation_model = FE(num_nodes=num_nodes, embedding_d=embedding_d)
        self.regression_model = regression_classes[reg_class](input_dim=embedding_d).to(self.device)

        
    def forward_embedding(self, indice_pairs):
        """
            indice_pairs: pairs of indices that we are interested in computting the embedding
                          it is a list of two lists, where the first list corresponds to the
                          first index and the second list corresponds to the second index of
                          corresponding pairs
        """

        # TODO:
        #      change this to a clearner solution to handle the list of indices versus tensor case
        z = self.representation_model()
        if isinstance(indice_pairs,list):
            z1 = z[indice_pairs[0]]
            z2 = z[indice_pairs[1]]
        else:
            z1 = z[indice_pairs[:,0]]
            z2 = z[indice_pairs[:,1]]
        return z1,z2 
        

    def predict(self, x, acquisition_fn='eps-greedy', if_dropout=False, return_var=False):
        """
            x: feature vector (should be of length (d+1)**2
            Returns the predicted loss.
        """
        if "GP" in self.reg_class:
            self.regression_model.eval()
            p = self.regression_model.predict(torch.hstack(x))
            if return_var:
                if len(p.mean.shape)>1:
                    p = p.mean.mean(dim=0), p.variance.mean(dim=0)
                else:
                    p = p.mean, p.variance
            else: 
                if len(p.mean.shape)>1:
                    p = p.mean.mean(dim=0)
                else:
                    p = p.mean
        else:
            if if_dropout:
                self.regression_model.train() # turn the dropout on
            else:
                self.regression_model.eval() # turn the dropout off
            if acquisition_fn=='eps-greedy':
                p = self.regression_model(*x)
            elif acquisition_fn=='variance':
                p = self.regression_model(*x)
            else:
                # unsupported acquisition_fn
                assert False
        return p
    
    def init_train(self):
        # To be consistent with ActiveGNN 
        return 
    
    def train_joint(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100, num_epoch_embedding=50, weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):
        print("Starting to train the joint model................................")
        train_x_indices = torch.Tensor(indice_pairs).long().T.to(self.device)
        # train_x = list(self.forward_embedding(self.data.x, train_x_indices))
        # self.regressor.set_train_data(inputs=train_x, targets=train_y.reshape(-1),strict=False)
        if self.reg_class=="Inner_product":
            dataset = TensorDataset(train_x_indices,train_y)
            data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
        else:
            print("Training the regression network..................................")
            self.regression_model.train() # turn the dropout on

            # for param in self.regression_model.parameters():
            #     param.requires_grad = True
            for param in self.regression_model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(self.regression_model.parameters(), lr=lr_reg, weight_decay=weight_decay_reg)
            for param in self.representation_model.parameters():
                param.requires_grad = False

            # scheduler = ExponentialLR(optimizer, gamma=gamma)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, verbose=True)

            dataset = TensorDataset(train_x_indices,train_y)
            data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
            for epoch in range(1,num_epoch+1):

                running_loss = 0
                for inner_iter, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
                    train_y_batch=train_y_batch.to(self.device)
                    optimizer.zero_grad()
                    y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
                    loss = F.mse_loss(y_pred_batch,train_y_batch) # regression loss
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                running_loss /= len(data_loader)
                # scheduler.step() # update the learning rate if stochastic training
                scheduler.step(running_loss) # for ReduceLROnPlateau scheduler

                # if epoch%5==1:
                # print(f'Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
        if self.update_embedding:
            print("Updating the embedding..................................")
            # self.regression_model.eval() # turn the dropout off

            # freeze the parameters of regression network
            for param in self.regression_model.parameters():
                param.requires_grad = False
            for param in self.representation_model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(self.representation_model.encoder.parameters(), lr=lr_embd, weight_decay=weight_decay_embd)
            # scheduler = ExponentialLR(optimizer, gamma=gamma)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, verbose=True)
            for epoch in range(1,num_epoch_embedding+1):

                running_loss = 0

                # training using only the regression loss
                for indice_pairs_batch, train_y_batch in data_loader:
                    train_y_batch = train_y_batch.reshape(-1).to(self.device)
                    optimizer.zero_grad()
                    y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
                    loss = F.mse_loss(y_pred_batch,train_y_batch) # regression loss

                    if self.lambda_ewc is not None:
                        # Add EWC penalty
                        for name, param in self.representation_model.encoder.named_parameters():
                            loss += self.lambda_ewc * torch.sum(torch.abs(self.encoder_fisher_information[name])*(param - self.encoder_opt_params[name]) ** 2)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                running_loss /= len(data_loader)
                # scheduler.step()
                scheduler.step(running_loss) # for ReduceLROnPlateau scheduler

                # if epoch%5==1:
                print(f'Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
        return
    
    def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9, ensemble_size=1):

        with torch.no_grad():
            y_norm = torch.mean(torch.abs(train_y.clone().detach())).item()
        self.regression_model.train() # turn the dropout on
        for param in self.regression_model.parameters():
            param.requires_grad = True
        if self.update_embedding: 
            for param in self.representation_model.parameters():
                param.requires_grad = True
                # optimizer = torch.optim.AdamW([
                #                 {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": 0 if self.reg_class=="Inner_product" else weight_decay_reg, "betas": (0.9, 0.999)},
                #                 {"params": self.representation_model.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
                optimizer = torch.optim.AdamW([
                                {"params": self.regression_model.parameters(), "lr": 0.001, "weight_decay": 0 if self.reg_class=="Inner_product" else weight_decay_reg, "betas": (0.9, 0.999)},
                                {"params": self.representation_model.parameters(),"lr": 0.001, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
        else:
            for param in self.representation_model.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(self.regression_model.parameters(), lr=lr_reg, weight_decay=weight_decay_embd)

        # print("optimizers are properly initalized", flush=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, min_lr=1e-12)

        dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
        data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=200, shuffle=False)
        best_loss =  torch.inf
        threshold = 1e-4
        lr_prev = [group["lr"] for group in optimizer.param_groups][0]
        # tol = 1e-6*math.sqrt(ensemble_size)
        for epoch in range(1,num_epoch+1):
            self.regression_model.train()
            # running_loss = 0
            for _, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
                train_y_batch = train_y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred_batch = self.regression_model(*self.forward_embedding(indice_pairs_batch)) 
                loss = F.huber_loss(y_pred_batch,train_y_batch)
                # torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(self.representation_model.parameters(), 1.0)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()
            # running_loss /= len(data_loader)

            self.regression_model.eval()  # Set model to evaluation mode

            total_loss = 0.0
            with torch.no_grad():  # No gradients needed
                # for _, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
                for _, (indice_pairs_batch, train_y_batch) in enumerate(val_data_loader):
                    train_y_batch = train_y_batch.to(self.device)
                    # print(f"updateing embedding {self.update_embedding}", flush=True)
                    y_pred_batch = self.regression_model(*self.forward_embedding(indice_pairs_batch)) 
                    loss = F.huber_loss(y_pred_batch,train_y_batch)
                    # loss = F.mse_loss(y_pred_batch,train_y_batch)
                    total_loss += loss.item() 
                total_loss /= len(data_loader)
            if total_loss<(1e-4*math.sqrt(ensemble_size)*y_norm):
                # print(f"stopped at epoch {epoch}", flush=True)
                break
            scheduler.step(total_loss)
            lr0 = scheduler._last_lr[0]
            if total_loss < best_loss*(1-threshold):
                best_reg_param = deepcopy(self.regression_model.state_dict())
                if self.update_embedding: 
                    best_gnn_param = deepcopy(self.representation_model.state_dict())
                best_loss = total_loss
            # scheduler.step(running_loss)
            # if running_loss < best_loss*(1-threshold):
            #     best_reg_param = deepcopy(self.regression_model.state_dict())
            #     if self.update_embedding: 
            #         best_gnn_param = deepcopy(self.representation_model.state_dict())
            #     best_loss = running_loss 
            if lr0<lr_prev:
                # print("learning rate decay, re-load from the best parameters", flush=True)
                self.regression_model.load_state_dict(best_reg_param)
                if self.update_embedding: 
                    self.representation_model.load_state_dict(best_gnn_param)
                lr_prev = lr0
            
            # if epoch%30==1:
            #     print(f'Device: {self.device}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
        self.regression_model.load_state_dict(best_reg_param)
        if self.update_embedding: 
            self.representation_model.load_state_dict(best_gnn_param)
        return
    