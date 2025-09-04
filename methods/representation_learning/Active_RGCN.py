import numpy as np
import math

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GAE

from methods.representation_learning.RGCN import RGCNEncoder, DistMultDecoder, negative_sampling
from methods.representation_learning.regression import DiagonalBilinearLayer, InnerProductLayer, BiLinearRegressionNetwork, RegressionNetwork


from copy import deepcopy

# import optuna


regression_classes = {"Linear": RegressionNetwork,
                      "BiLinear_diagonal": DiagonalBilinearLayer, 
                      "Inner_product": InnerProductLayer, 
                      "Bilinear": BiLinearRegressionNetwork,
                    }
                    
class ActiveGNN(torch.nn.Module):
    def __init__(self, data, node_dim, update_embedding=True, num_epoch_init=300,
                 embedding_d=50, lambda_ewc=None, hiv_indices=None, hidden_dim=128,
                 num_conv_layers=1, reg_class="Bilinear", device="cpu"):
        """

        """
        super().__init__()
        self.device = device 

        self.update_embedding = update_embedding
        self.data = data.to(self.device)
        self.data.x = self.data.x.to(self.device)
        self.data.edge_index = self.data.edge_index.to(self.device)
        self.data.edge_type = self.data.edge_type.to(self.device)
        self.loader = None 
        
        # TODO: implement neighbor sampling 
        # if hiv_indices is None:
        # else: 
        #     self.loader = NeighborLoader(
        #         self.data,
        #         input_nodes=torch.tensor(hiv_indices),
        #         num_neighbors=[10,10],
        #         batch_size=len(hiv_indices),
        #         replace=False,
        #         shuffle=False,
        #     )
        self.lambda_ewc = lambda_ewc # strength of penalty on how
                                     # much the embedding is allowed
                                     # to change
        self.hiv_indices = hiv_indices
        self.reg_class = reg_class
        self.input_dim = embedding_d

        # memory_init = torch.cuda.memory_allocated(self.device)
        # self.data.to(self.device)
        # memory_data_added = torch.cuda.memory_allocated(self.device)
        # print("data is using {} fytes of memory".format(memory_data_added-memory_init))
        self.num_relations = len(set(data.edge_type.tolist()))

        
        self.representation_model = GAE(
            RGCNEncoder(data.num_nodes, node_dim, embedding_d, self.num_relations, hidden_dim, num_conv_layers, device=self.device),
            DistMultDecoder(self.num_relations, embedding_d, device=self.device),
        ).to(self.device)


        self.nparam_encoder = np.sum([param.numel() for param in self.representation_model.encoder.parameters() if param.requires_grad])
        self.nparam_decoder = np.sum([param.numel() for param in self.representation_model.decoder.parameters() if param.requires_grad])
        self.nparam_model = self.nparam_decoder+ self.nparam_encoder

        self.encoder_opt_params = None
        self.encoder_opt_params_dict = None
        self.encoder_fisher_information = None

        self.num_epoch_init = num_epoch_init
        self.regression_model = regression_classes[reg_class](input_dim=embedding_d).to(self.device)

        # re-weight the loss for already seen data 
        # default to be 1 so do not affect the training
        self.loss_reweight = 1
        
    def forward_embedding(self, x, indice_pairs):
        """
            indice_pairs: pairs of indices that we are interested in computting the embedding
                          it is a list of two lists, where the first list corresponds to the
                          first index and the second list corresponds to the second index of
                          corresponding pairs
        """

        # TODO:
        #      change this to a clearner solution to handle the list of indices versus tensor case
        z = self.representation_model.encode(x, self.data.edge_index, self.data.edge_type)
        if self.hiv_indices is not None:
            z = z[self.hiv_indices]
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

    def edge_train_test_split(self):
        if self.loader is None:
            alledge = list(range(len(self.data.edge_type)))
            train_i, test_i = train_test_split(alledge, test_size=0.2, shuffle=True)

            train_edge_index= self.data.edge_index[:,train_i].tolist()
            train_edge_type= self.data.edge_type[train_i].tolist()

            test_edge_index= self.data.edge_index[:,test_i].tolist()
            test_edge_type= self.data.edge_type[test_i].tolist()

            self.data.train_edge_index = torch.tensor(train_edge_index)
            self.data.train_edge_type = torch.tensor(train_edge_type)
            self.data.test_edge_index =torch.tensor(test_edge_index)
            self.data.test_edge_type = torch.tensor(test_edge_type)
            self.data.to(self.device)
        return
    
    def init_train(self):
        """
            Initialize the GNN using encoder-decoder negative sampling 
        """
        if self.loader is None:
            self.edge_train_test_split()
            optimizer = torch.optim.RMSprop(self.representation_model.parameters(), lr=0.001)
            # optimizer = torch.optim.AdamW(self.representation_model.parameters(), lr=0.01, weight_decay=1e-5)

            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
            # print("starting to train the inital embedding................................")
            for epoch in range(1,self.num_epoch_init+1):
                optimizer.zero_grad()
                # z = self.model.encode(self.data.edge_index, self.data.edge_type)
                z = self.representation_model.encode(self.data.x, self.data.edge_index, self.data.edge_type)
                pos_out = self.representation_model.decode(z, self.data.train_edge_index, self.data.train_edge_type)

                neg_edge_index = negative_sampling(self.data.train_edge_index, self.data.num_nodes)
                neg_out = self.representation_model.decode(z, neg_edge_index, self.data.train_edge_type)

                out = torch.cat([pos_out, neg_out])
                gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
                pen_loss = z.pow(2).mean() + self.representation_model.decoder.rel_emb.pow(2).mean() # penalty loss
                loss = cross_entropy_loss + 1e-2 * pen_loss

                loss.backward()
                
                optimizer.step()
                # scheduler.step(loss.item())
                # if epoch%20==1:
                #     print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
            if self.lambda_ewc is not None:
                fisher_information = {}
                for name, param in self.representation_model.encoder.named_parameters():
                    fisher_information[name] = torch.zeros_like(param)

                self.representation_model.zero_grad()
                z = self.representation_model.encode(self.data.x, self.data.edge_index, self.data.edge_type)
                pos_out = self.representation_model.decode(z, self.data.train_edge_index, self.data.train_edge_type)

                neg_edge_index = negative_sampling(self.data.train_edge_index, self.data.num_nodes)
                neg_out = self.representation_model.decode(z, neg_edge_index, self.data.train_edge_type)

                out = torch.cat([pos_out, neg_out])
                gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                entropy_loss = F.binary_cross_entropy_with_logits(out, gt) # for the purpose of grad this is equiavalent to nll loss

                grad_params = torch.autograd.grad(entropy_loss, self.representation_model.encoder.parameters(), create_graph=True)
                for _ in range(1000):
                    # Generate random vectors for each parameter
                    random_vectors = [torch.randn_like(param) for param in self.representation_model.encoder.parameters()]
                    
                    # Compute Hessian-vector product for each parameter
                    hvp = torch.autograd.grad(
                        outputs=grad_params,
                        inputs=self.representation_model.encoder.parameters(),
                        grad_outputs=random_vectors,
                        retain_graph=True
                    )
                    
                    # Update the estimate of the diagonal
                    for (name, _), v, hv in zip(self.representation_model.encoder.named_parameters(), random_vectors, hvp):
                        fisher_information[name] += v*hv

                for name, param in self.representation_model.encoder.named_parameters():
                    # Average the estimates
                    fisher_information[name] = fisher_information[name] / 1000

                ## empirical fisher but too slow due to large for loop 
                # for pred, label in zip(out,gt):
                #     optimizer.zero_grad()
                #     entropy_loss= F.binary_cross_entropy_with_logits(pred, label) # for the purpose of grad this is equiavalent to nll loss
                #     entropy_loss.backward(retain_graph=True)
                #     for name, param in self.representation_model.encoder.named_parameters():
                #         # fisher_information[name] += param.grad.detach().clone() ** 2 / len(gt) # the empirical estimate of the (diagonal) fisher information matrix (square-loss)
                #         fisher_information[name] += param.grad.detach().clone()**2*(torch.sigmoid(pred.detach().clone())*(1-torch.sigmoid(pred.detach().clone()))) / len(gt) # the empirical estimate of the (diagonal) fisher information matrix (cross entropy loss)
                self.encoder_fisher_information = fisher_information
                self.encoder_opt_params = {name: param.detach().clone() for name, param in self.representation_model.encoder.named_parameters()}
        else:
            # gc.collect()
            # torch.cuda.empty_cache() 
            optimizer = torch.optim.RMSprop(self.representation_model.parameters(), lr=0.001)
            # optimizer = torch.optim.AdamW(self.representation_model.parameters(), lr=0.001, weight_decay=1e-5)

            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
            print("starting to train the inital embedding................................")
            for epoch in range(1,self.num_epoch_init+1):
                for data_batch in self.loader:
                    optimizer.zero_grad()
                    alledge = list(range(len(data_batch.edge_type)))
                    train_i, test_i = train_test_split(alledge, test_size=0.2, shuffle=True)

                    train_edge_index= data_batch.edge_index[:,train_i].tolist()
                    train_edge_type= data_batch.edge_type[train_i].tolist()

                    test_edge_index= data_batch.edge_index[:,test_i].tolist()
                    test_edge_type= data_batch.edge_type[test_i].tolist()

                    data_batch.train_edge_index = torch.tensor(train_edge_index)
                    data_batch.train_edge_type = torch.tensor(train_edge_type)
                    data_batch.test_edge_index =torch.tensor(test_edge_index)
                    data_batch.test_edge_type = torch.tensor(test_edge_type)
                    data_batch.to(self.device)
                    
                    # z = self.model.encode(data_batch.test_edge_index, data_batch.test_edge_type)
                    z = self.representation_model.encode(data_batch.x, data_batch.edge_index, data_batch.edge_type)
                    pos_out = self.representation_model.decode(z, data_batch.train_edge_index, data_batch.train_edge_type)

                    neg_edge_index = negative_sampling(data_batch.train_edge_index, data_batch.num_nodes)
                    neg_out = self.representation_model.decode(z, neg_edge_index, data_batch.train_edge_type)

                    out = torch.cat([pos_out, neg_out])
                    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
                    pen_loss = z.pow(2).mean() + self.representation_model.decoder.rel_emb.pow(2).mean() # penalty loss
                    loss = cross_entropy_loss + 1e-2 * pen_loss

                    loss.backward()
                    
                    optimizer.step()
                    running_loss += loss.item()
                running_loss /= len(self.loader)
                # scheduler.step(running_loss)
                # if epoch%2==1:
                #     print(f'Epoch: {epoch:05d}, Loss: {running_loss:.4f}')
            if self.lambda_ewc is not None:
                fisher_information = {}
                for name, param in self.representation_model.encoder.named_parameters():
                    fisher_information[name] = torch.zeros_like(param)

                for data_batch in self.loader:
                    self.representation_model.zero_grad()
                    alledge = list(range(len(data_batch.edge_type)))
                    train_i, test_i = train_test_split(alledge, test_size=0.2, shuffle=True)

                    train_edge_index= data_batch.edge_index[:,train_i].tolist()
                    train_edge_type= data_batch.edge_type[train_i].tolist()

                    test_edge_index= data_batch.edge_index[:,test_i].tolist()
                    test_edge_type= data_batch.edge_type[test_i].tolist()

                    data_batch.train_edge_index = torch.tensor(train_edge_index)
                    data_batch.train_edge_type = torch.tensor(train_edge_type)
                    data_batch.test_edge_index =torch.tensor(test_edge_index)
                    data_batch.test_edge_type = torch.tensor(test_edge_type)
                    data_batch.to(self.device)
                
                    z = self.representation_model.encode(data_batch.x, data_batch.edge_index, data_batch.edge_type)
                    pos_out = self.representation_model.decode(z, data_batch.train_edge_index, data_batch.train_edge_type)

                    neg_edge_index = negative_sampling(data_batch.train_edge_index, data_batch.num_nodes)
                    neg_out = self.representation_model.decode(z, neg_edge_index, data_batch.train_edge_type)

                    out = torch.cat([pos_out, neg_out])
                    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                    for pred, label in zip(out,gt):
                        optimizer.zero_grad()
                        entropy_loss= F.binary_cross_entropy_with_logits(pred, label) # for the purpose of grad this is equiavalent to nll loss
                        entropy_loss.backward(retain_graph=True)
                        for name, param in self.representation_model.encoder.named_parameters():
                            # fisher_information[name] += param.grad.detach().clone() ** 2 / len(gt) # the empirical estimate of the (diagonal) fisher information matrix (square-loss)
                            fisher_information[name] += param.grad.detach().clone()**2*(torch.sigmoid(pred.detach().clone())*(1-torch.sigmoid(pred.detach().clone()))) / len(gt) # the empirical estimate of the (diagonal) fisher information matrix (cross entropy loss)
                self.encoder_fisher_information = fisher_information
                self.encoder_opt_params = {name: param.detach().clone() for name, param in self.representation_model.encoder.named_parameters()}
        self.encoder_opt_params_dict = deepcopy(self.representation_model.encoder.state_dict())
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

            # freeze the parameters of encoder
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
                print(f'Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
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
    
    def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9, embeddings=None, ensemble_size=1):
        with torch.no_grad():
            y_norm = torch.mean(torch.abs(train_y.clone().detach())).item()
        # params_dicts = {0: {'lr_reg': 0.00011775633539255192, 'lr_embd': 0.0002782704109955827, 'patience': 5, 'factor': 0.12744063613204343}, 
        #                 1: {'lr_reg': 7.851480146099328e-05, 'lr_embd': 0.0002562500105605308, 'patience': 18, 'factor': 0.944286553072319}, 
        #                 2: {'lr_reg': 0.0004334540230411259, 'lr_embd': 0.000164313999678611, 'patience': 7, 'factor': 0.5176930418639185}, 
        #                 3: {'lr_reg': 2.893794327968638e-05, 'lr_embd': 0.00032526352200831814, 'patience': 9, 'factor': 0.6948224576496399}}
        # params = params_dicts[self.device.index]
        if self.reg_class == "Bilinear":
            # if num_epoch<100:
            #     params = {'lr_reg': 0.0011302523394573292, 'lr_embd': 0.0009819128690268793, 'patience': 7, 'factor': 0.4482218043946298}
            # else:
                # params = {'lr_reg': 0.00011775633539255192, 'lr_embd': 0.0002782704109955827, 'patience': 5, 'factor': 0.12744063613204343}
                # params = {'lr_reg': lr_reg, 'lr_embd': lr_embd, 'patience': 10, 'factor': 0.1}
            # params = {'lr_reg': 0.001, 'lr_embd': 0.001}
            params = {'lr_reg': 0.01, 'lr_embd': 1e-3}
        if self.reg_class == "BiLinear_diagonal":
            params = {'lr_reg': 0.025182743804349022, 'lr_embd': 0.001535788244157919, 'patience': 5, 'factor': 0.25399203546253607}
        if self.reg_class =="Inner_product":
            params = {'lr_reg': 0.0007826737053612429, 'lr_embd': 0.000348543389678371, 'patience': 8, 'factor': 0.6710017805115978}
        # if lr_embd==1e-3:
        #     print("Only updating the embeddings................................",flush=True)
        #     for param in self.regression_model.parameters():
        #         param.requires_grad = False 
        # else:
        # print("Updating the joint model................................",flush=True)
        for param in self.regression_model.parameters():
            param.requires_grad = True
        if self.update_embedding: 
            for param in self.representation_model.encoder.parameters():
                param.requires_grad = True
            # if lr_embd==1e-3:
            # optimizer = torch.optim.AdamW(self.representation_model.parameters(), lr=lr_embd, weight_decay=weight_decay_reg)
            # else:
            optimizer = torch.optim.AdamW([
                            {"params": self.regression_model.parameters(), "lr": params["lr_reg"], "weight_decay": 0 if self.reg_class=="Inner_product" else weight_decay_reg, "betas": (0.9, 0.999)},
                            {"params": self.representation_model.encoder.parameters(),"lr": params["lr_embd"], "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
        else:
            # print("only updating the regression models", flush=True)
            for param in self.representation_model.encoder.parameters():
                param.requires_grad = False
            for param in self.regression_model.parameters():
                param.requires_grad = True
            # optimizer = torch.optim.AdamW(self.regression_model.parameters(), lr=lr_reg, weight_decay=weight_decay_embd)
            optimizer = torch.optim.AdamW(self.regression_model.parameters(), lr=0.01, weight_decay=weight_decay_reg)

        # print("optimizers are properly initalized", flush=True)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["factor"], patience=params["patience"], verbose=True, min_lr=1e-12)
        # optimizer = torch.optim.Adam(list(self.regressor.parameters())+list(self.model.parameters()), lr=lr, weight_decay=1e-2)
        # if lr_embd==1e-3:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, min_lr=1e-12)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, verbose=True, min_lr=1e-12)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=20, verbose=True, min_lr=1e-12)
        # else:
            # scheduler = ExponentialLR(optimizer, gamma=0.9)

        dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
        # print("batch size 50",flush=True)
        data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=200, shuffle=False)
        # data_loader = DataLoader(dataset, batch_size=100, shuffle=True)
        # data_loader = DataLoader(dataset, batch_size=200, shuffle=True)
        # data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
        # else:
        #     # data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
        #     # data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        #     print("batch size 100",flush=True)
        #     data_loader = DataLoader(dataset, batch_size=100, shuffle=True)
        best_loss =  torch.inf
        # prev_loss = torch.inf 
        threshold = 1e-4
        lr_prev = [group["lr"] for group in optimizer.param_groups][0]
        # tol = 1e-6*math.sqrt(ensemble_size)
        # # progress = 0.99 # for SophiaG
        # for epoch in range(1,51):
        for epoch in range(1,num_epoch+1):
            self.regression_model.train() # turn the dropout on

            # running_loss = 0
            for _, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
                train_y_batch = train_y_batch.to(self.device)
                optimizer.zero_grad()
                if self.update_embedding:
                    y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
                else:
                    if isinstance(indice_pairs_batch,list):
                        z1 = embeddings[indice_pairs_batch[0]]
                        z2 = embeddings[indice_pairs_batch[1]]
                    else:
                        z1 = embeddings[indice_pairs_batch[:,0]]
                        z2 = embeddings[indice_pairs_batch[:,1]]
                    y_pred_batch = self.regression_model(z1,z2)
                loss = F.huber_loss(y_pred_batch,train_y_batch)
                # loss = F.mse_loss(y_pred_batch,train_y_batch)
                loss.backward()
                # if not self.update_embedding: 
                #     torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
                #     torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
                optimizer.step()
                # running_loss += loss.item()
            # running_loss /= len(data_loader)
            # scheduler.step(running_loss)
            # if lr_embd==1e-3:
            # if epoch==10:
            #     for param in self.regression_model.parameters():
            #         param.requires_grad = True
            #     optimizer = torch.optim.AdamW([
            #                 {"params": self.regression_model.parameters(), "lr": params["lr_reg"], "weight_decay": 0 if self.reg_class=="Inner_product" else weight_decay_reg, "betas": (0.9, 0.999)},
            #                 {"params": self.representation_model.encoder.parameters(),"lr": params["lr_embd"], "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
            #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, verbose=True, min_lr=1e-12)
            # if epoch>10:

            self.regression_model.eval()  # Set model to evaluation mode
            total_loss = 0.0
            with torch.no_grad():  # No gradients needed
                # for _, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
                for _, (indice_pairs_batch, train_y_batch) in enumerate(val_data_loader):
                    train_y_batch = train_y_batch.to(self.device)
                    # print(f"updateing embedding {self.update_embedding}", flush=True)
                    if self.update_embedding:
                        y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
                    else:
                        if isinstance(indice_pairs_batch,list):
                            z1 = embeddings[indice_pairs_batch[0]]
                            z2 = embeddings[indice_pairs_batch[1]]
                        else:
                            z1 = embeddings[indice_pairs_batch[:,0]]
                            z2 = embeddings[indice_pairs_batch[:,1]]
                        y_pred_batch = self.regression_model(z1,z2)
                    loss = F.huber_loss(y_pred_batch,train_y_batch)
                    # loss = F.mse_loss(y_pred_batch,train_y_batch)
                    total_loss += loss.item() 
                total_loss /= len(data_loader)
            scheduler.step(total_loss)
            # else:
                # scheduler.step()
            lr0 = scheduler._last_lr[0]
            if total_loss < best_loss*(1-threshold):
                best_reg_param = deepcopy(self.regression_model.state_dict())
                if self.update_embedding: 
                    best_gnn_param = deepcopy(self.representation_model.state_dict())
                best_loss = total_loss
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
            # if lr0 < tol:
            #     print(f"stopped at epoch {epoch}", flush=True)
            #     break
            if total_loss<(1e-4*math.sqrt(ensemble_size)*y_norm):
                # print(f"stopped at epoch {epoch}", flush=True)
                break
            # if np.abs(running_loss-prev_loss)<(1e-5*math.sqrt(ensemble_size)):
            #     print(f"stopped at epoch {epoch}", flush=True)
            #     break
            # prev_loss = running_loss
            # if epoch%30==1:
            #     print(f'Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
                # print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
        # print(running_loss, flush=True)
        # print(prev_loss, flush=True)
        # print(total_loss, flush=True)
        # print(y_norm, flush=True)
        self.regression_model.load_state_dict(best_reg_param)
        if self.update_embedding: 
            self.representation_model.load_state_dict(best_gnn_param)
        return
    

    # def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):
    #     # parameters for non-stochastic training for bilinear models
    #     if self.reg_class == "Bilinear":
    #         params = {'lr_reg': 0.009229902617612173, 'lr_embd': 0.004789794758681398, 'patience': 182, 'factor': 0.8730759123690306}
    #     if  self.reg_class == "BiLinear_diagonal":
    #         params = {'lr_reg': 0.002538934609750822, 'lr_embd': 0.0077205270821279785, 'patience': 174, 'factor': 0.9366051537091166}

    #     self.regression_model.train() # turn the dropout on
    #     for param in self.regression_model.parameters():
    #         param.requires_grad = True
    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True
    #     print("Starting to train the joint model................................")

    #     optimizer = torch.optim.AdamW([
    #                     {"params": self.regression_model.parameters(), "lr": params["lr_reg"], "weight_decay": weight_decay_reg, "betas": (0.9, 0.999)},
    #                     {"params": self.representation_model.encoder.parameters(),"lr": params["lr_embd"], "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])

    #     print("optimizers are properly initalized", flush=True)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["factor"], patience=params["patience"], verbose=True, min_lr=1e-12)

    #     indices = torch.Tensor(indice_pairs).long().T.to(self.device)
    #     train_y = train_y.to(self.device)
    #     best_loss =  torch.inf
    #     threshold = 1e-4
    #     for epoch in range(1,num_epoch+1):

    #         optimizer.zero_grad()
    #         y_pred = self.regression_model(*self.forward_embedding(self.data.x, indices)) 
    #         loss = F.huber_loss(y_pred,train_y)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
    #         torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #         optimizer.step()
    #         running_loss = loss.item()
    #         scheduler.step(running_loss)
    #         if running_loss < best_loss*(1-threshold):
    #             best_reg_param = deepcopy(self.regression_model.state_dict())
    #             best_gnn_param = deepcopy(self.representation_model.state_dict())
    #             best_loss = running_loss 
    #         if epoch%1000==1:
    #             print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
    #         if running_loss < 1e-4:
    #             break
    #     self.regression_model.load_state_dict(best_reg_param)
    #     self.representation_model.load_state_dict(best_gnn_param)
    #     return

    # def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):
        
    #     params_dicts = {0: {'lr': 0.00026155780123332675, 'patience': 10, 'factor': 0.770632876806435}, 
    #                     1: {'lr': 0.00031188301485156024, 'patience': 18, 'factor': 0.8858278851434156}, 
    #                     2: {'lr': 0.0002863922808066983, 'patience': 19, 'factor': 0.16107719761483025}, 
    #                     3: {'lr': 0.00031490499301877877, 'patience': 9, 'factor': 0.8028189205918379}}
    #     params = params_dicts[self.device.index]

    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True
    #     print("Starting to train the GNN model................................")

    #     optimizer = torch.optim.AdamW(self.representation_model.encoder.parameters(),lr=params["lr"], weight_decay=weight_decay_embd, betas=(0.9, 0.99))

    #     print("optimizers are properly initalized", flush=True)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["factor"], patience=params["patience"], verbose=True, min_lr=1e-12)

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     data_loader = DataLoader(dataset, batch_size=100)
    #     # data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     best_loss =  torch.inf
    #     # # progress = 0.99 # for SophiaG
    #     for epoch in range(1,num_epoch+1):

    #         running_loss = 0
    #         # mse_loss = 0
    #         hubber_loss = 0
    #         for _, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             optimizer.zero_grad()
    #             y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
    #             # loss = F.huber_loss(y_pred_batch,train_y_batch)
    #             loss = F.mse_loss(y_pred_batch,train_y_batch).item()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #             optimizer.step()
    #             running_loss += loss.item()                
    #             # mse_loss += F.mse_loss(y_pred_batch,train_y_batch).item()
    #             hubber_loss += F.huber_loss(y_pred_batch,train_y_batch)
    #         running_loss /= len(data_loader)
    #         mse_loss /= len(data_loader)            
    #         scheduler.step(running_loss)

    #         y_pred = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs)) 
    #         hubber_epoch_loss = F.huber_loss(y_pred, train_y.to(self.device)).detach()  
    #         mse_epoch_loss = F.mse_loss(y_pred, train_y.to(self.device)).detach()

    #         if running_loss < best_loss*(0.999):
    #             best_gnn_param = deepcopy(self.representation_model.encoder.state_dict())
    #             best_loss = running_loss
    #         # if epoch%5==1:
    #         print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}, MSE: {mse_loss:.4f}, Epoch Loss: {hubber_epoch_loss:.4f}, Epoch MSE: {mse_epoch_loss:.4f}', flush=True)
    #     self.representation_model.encoder.load_state_dict(best_gnn_param)
    #     return


    # def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):
        
    #     params_dicts = {0: {'lr': 0.007579003351579659, 'patience': 189, 'factor': 0.5761959068827369}, 
    #                     1: {'lr': 0.006497739783521893, 'patience': 186, 'factor': 0.9026263133448438}, 
    #                     2: {'lr': 0.003653095435501275, 'patience': 178, 'factor': 0.45172139651205995}, 
    #                     3: {'lr': 0.0067176392217949705, 'patience': 179, 'factor': 0.903632096010776}
    #                     }
    #     params = params_dicts[self.device.index]

    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True
    #     print("Starting to train the GNN model................................")

    #     optimizer = torch.optim.AdamW(self.representation_model.encoder.parameters(),lr=params["lr"], weight_decay=weight_decay_embd, betas=(0.9, 0.99))

    #     print("optimizers are properly initalized", flush=True)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["factor"], patience=params["patience"], verbose=True, min_lr=1e-12)

    #     indices = torch.Tensor(indice_pairs).long().T.to(self.device)
    #     best_loss =  torch.inf
    #     for epoch in range(1,num_epoch+1):

    #         optimizer.zero_grad()
    #         y_pred = self.regression_model(*self.forward_embedding(self.data.x, indices)) 
    #         loss = F.huber_loss(y_pred,train_y.to(self.device))
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #         optimizer.step()
    #         running_loss = loss.item()
    #         scheduler.step(running_loss)
    #         mse_epoch_loss = F.mse_loss(y_pred, train_y.to(self.device)).detach()

    #         if running_loss < best_loss*(0.999):
    #             best_gnn_param = deepcopy(self.representation_model.encoder.state_dict())
    #             best_loss = running_loss
    #         # if running_loss < 1e-6:
    #             # break
    #         if epoch%1000==1:
    #             print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}, Epoch MSE: {mse_epoch_loss:.4f}', flush=True)
    #     self.representation_model.encoder.load_state_dict(best_gnn_param)
    #     return
    
    # def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):

    #     params_dicts = {
    #         0: {'lr': 3.6682598280391346e-05,  'patience': 7, 'factor': 0.1},
    #         1: {'lr': 3.6682598280391346e-05,  'patience': 7, 'factor': 0.1},
    #         2: {'lr': 3.6682598280391346e-05,  'patience': 7, 'factor': 0.1},
    #         3: {'lr': 3.6682598280391346e-05,  'patience': 7, 'factor': 0.1}
    #     }
    #     params = params_dicts[self.device.index]
        
    #     print("Starting to train the embedding model................................")
    #     for param in self.representation_model.parameters():
    #         param.requires_grad = True

    #     #inner product regression model so no training paramters in the regression network 
    #     optimizer = torch.optim.AdamW(self.representation_model.parameters(), lr=params["lr"], weight_decay=weight_decay_embd)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["factor"], patience=params["patience"], verbose=True, min_lr=1e-12)

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     best_loss =  float('inf')

    #     threshold = 1e-4
    #     for epoch in range(1,num_epoch+1):

    #         running_loss = 0
    #         for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             optimizer.zero_grad()
    #             y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
    #             loss = F.huber_loss(y_pred_batch,train_y_batch)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0, norm_type="inf")
    #             optimizer.step()
    #             running_loss += loss
    #         running_loss /= len(data_loader)
    #         scheduler.step(running_loss) # for ReduceLROnPlateau scheduler
    #         if running_loss < best_loss*(1-threshold):
    #             best_reg_param = self.regression_model.state_dict()
    #             best_gnn_param = self.representation_model.state_dict()
    #             best_loss = running_loss 
    #         print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
    #     self.regression_model.load_state_dict(best_reg_param)
    #     self.representation_model.load_state_dict(best_gnn_param)
    #     return


    # def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):
    #     params_dicts = {
    #         0: {'lr_reg': 0.0007801743354238085, 'lr_embd': 0.0006502636328427544, 'T0': 148, 'patience': 9, 'cooldown': 3},
    #         1: {'lr_reg': 0.0007801743354238085, 'lr_embd': 0.0006502636328427544, 'T0': 148, 'patience': 9, 'cooldown': 3},
    #         2: {'lr_reg': 0.0007801743354238085, 'lr_embd': 0.0006502636328427544, 'T0': 148, 'patience': 9, 'cooldown': 3},
    #         3: {'lr_reg': 0.0007801743354238085, 'lr_embd': 0.0006502636328427544, 'T0': 148, 'patience': 9, 'cooldown': 3}
    #     }
    #     params = params_dicts[self.device.index]
        
    #     self.regression_model.train() # turn the dropout on
    #     for param in self.regression_model.parameters():
    #             param.requires_grad = True 

    #     for param in self.representation_model.parameters():
    #         param.requires_grad = True
    #     print("Starting to train the joint model................................")

    #     optimizer = SwitchableOptimizer (list(self.regression_model.parameters())+list(self.representation_model.encoder.parameters()), lr_2nd=params["lr_2nd"], lr_1st=params["lr_1st"]) # cosine scheduler

    #     print("optimizers are properly initalized", flush=True)
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=params["T0"])

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     best_loss =  float('inf')
    #     running_loss_prev = torch.inf 

    #     patience = params["patience"]
    #     patience_ref = 0
    #     threshold = 1e-4
    #     progress = 0.999 
    #     cooldown = params["cooldown"]
    #     cooldown_counter = cooldown
    #     shrink_lr = False 
    #     for epoch in range(1,num_epoch+1):

    #         running_loss = 0
    #         for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             optimizer.zero_grad()
    #             y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
    #             loss = F.huber_loss(y_pred_batch,train_y_batch)
    #             loss.backward()
                
    #             torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
    #             torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #             optimizer.step()
    #             scheduler.step()
    #             if shrink_lr and (scheduler.get_last_lr()[0]==scheduler.base_lrs[0]):
    #                 scheduler.base_lrs[:] = [lr*params["factor"] for lr in scheduler.base_lrs]
    #                 shrink_lr = False 
    #             running_loss += loss
    #         running_loss /= len(data_loader)
    #         # second_order = (running_loss>running_loss_prev)
    #         # running_loss_prev = running_loss
    #         if running_loss < best_loss*(1-threshold):
    #             best_reg_param = self.regression_model.state_dict()
    #             best_gnn_param = self.representation_model.state_dict()
    #             best_loss = running_loss 
    #         else: 
    #             if (running_loss > running_loss_prev*progress) and (not shrink_lr):
    #                 if cooldown_counter:
    #                     cooldown_counter-=1 
    #                 else:
    #                     patience_ref += 1
    #             if patience_ref >= patience:
    #                 shrink_lr = True 
    #                 patience_ref = 0
    #                 cooldown_counter = cooldown
    #         running_loss_prev = running_loss
    #         print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
    #     self.regression_model.load_state_dict(best_reg_param)
    #     self.representation_model.load_state_dict(best_gnn_param)
    #     return

    # def train_joint_non_alternating(self, train_y, indice_pairs, lr_reg=0.001, lr_embd=0.001, num_epoch=100,  weight_decay_reg=1e-2, weight_decay_embd=1e-2, gamma=0.9):

    #     # params_dicts = {
    #     #     0: {'lr_1st': 3.6682598280391346e-05, 'lr_2nd': 0.3813662256523944, 'patience': 7, 'cooldown': 3},
    #     #     1: {'lr_1st': 2.252501842800183e-05, 'lr_2nd': 0.011562860771751053, 'patience': 9, 'cooldown': 0},
    #     #     2: {'lr_1st': 2.5647541300353692e-05, 'lr_2nd': 0.00589723468037325, 'patience': 4, 'cooldown': 0},
    #     #     3: {'lr_1st': 3e-5, 'lr_2nd': 0.01, 'patience': 5, 'cooldown': 0}
    #     # }
    #     # params_dicts = {
    #     #     0: {'lr_1st': 5e-5, 'lr_2nd': 1, 'patience': 6, 'cooldown': 3},
    #     #     1: {'lr_1st': 5e-5, 'lr_2nd': 0.5, 'patience': 6, 'cooldown': 3},
    #     #     2: {'lr_1st': 5e-5, 'lr_2nd': 0.05, 'patience': 6, 'cooldown': 3},
    #     #     3: {'lr_1st': 4.175809005654464e-05, 'lr_2nd': 0.0027468494475949602, 'patience': 6, 'cooldown': 3}
    #     # }
    #     # params_dicts = {
    #     #     0: {'lr_1st': 5.5961837505411256e-05, 'lr_2nd': 0.03949302983881044, 'patience': 5, 'cooldown': 5},
    #     #     1: {'lr_1st': 5e-5, 'lr_2nd': 0.5, 'patience': 6, 'cooldown': 3},
    #     #     2: {'lr_1st': 5e-5, 'lr_2nd': 0.05, 'patience': 6, 'cooldown': 3},
    #     #     3: {'lr_1st': 4.175809005654464e-05, 'lr_2nd': 1, 'patience': 5, 'cooldown': 5}
    #     # }
    #     # params = params_dicts[self.device.index]
        
    #     print("Starting to train the joint model................................")
    #     if self.reg_class=="DeepGP":
    #         mll = DeepApproximateMLL(VariationalELBO(self.regression_model.likelihood, self.regression_model, len(train_y)))
    #     elif self.reg_class=="ExactLinkGP":
    #         mll = ExactMarginalLogLikelihood(self.regression_model.likelihood, self.regression_model)
    #     elif self.reg_class=="ApprxGP":
    #         mll = VariationalELBO(self.regression_model.likelihood, self.regression_model, len(train_y))
    #     self.regression_model.train() # turn the dropout on
    #     if self.update_embedding:
    #         for param in self.representation_model.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in self.representation_model.parameters():
    #             param.requires_grad = False

    #     if sum(p.numel() for p in self.regression_model.parameters())==0: 
    #         #inner product regression model so no training paramters in the regression network 
    #         optimizer = torch.optim.AdamW(self.representation_model.parameters(), lr=lr_embd, weight_decay=weight_decay_embd)
    #     elif self.update_embedding:
    #         # optimizer = torch.optim.AdamW(self.representation_model.parameters(), lr=lr_embd, weight_decay=weight_decay)
    #         # optimizer = torch.optim.AdamW([
    #         #                 {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": weight_decay_reg, "betas": (0.9, 0.999)},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
    #         # optimizer = Lion([
    #         #                 {"params": self.regression_model.parameters(), "lr": lr_reg/5, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": lr_embd/5, "weight_decay": weight_decay_embd}])

    #         # optimizer = Lion([
    #         #                 {"params": self.regression_model.parameters(), "lr": 1e-3, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": 1e-4, "weight_decay": weight_decay_embd}])

    #         # optimizer = Lion([
    #         #                 {"params": self.regression_model.parameters(), "lr": 1e-4, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": 1e-4, "weight_decay": weight_decay_embd}])

    #         # optimizer = Lion([
    #         #                 {"params": self.regression_model.parameters(), "lr": 1e-4, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": 1e-5, "weight_decay": weight_decay_embd}])

    #         # optimizer = Lion([
    #         #                 {"params": self.regression_model.parameters(), "lr": 1e-3, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": 5e-4, "weight_decay": weight_decay_embd}])

    #         # optimizer = Lion([
    #         #                 {"params": self.regression_model.parameters(), "lr": 1e-4, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": 5e-5, "weight_decay": weight_decay_embd}])
            
    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": lr_reg/5, "weight_decay": weight_decay_reg},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": lr_embd/5, "weight_decay": weight_decay_embd}])

    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": 1e-3, "weight_decay": weight_decay_reg},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": 1e-4, "weight_decay": weight_decay_embd}])

    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": 1e-3, "weight_decay": weight_decay_reg},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": 5e-4, "weight_decay": weight_decay_embd}])

    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": 1e-4, "weight_decay": weight_decay_reg, "rho": 0.1},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": 5e-5, "weight_decay": weight_decay_embd, "rho": 0.1}])

    #         # optimizer = SwitchableOptimizer (list(self.regression_model.parameters())+list(self.representation_model.encoder.parameters()), threshold_lr=5e-5, initial_lr=1e-3, patience=5)
    #         optimizer = SwitchableOptimizer (list(self.regression_model.parameters())+list(self.representation_model.encoder.parameters()), lr_2nd=params["lr_2nd"], lr_1st=params["lr_1st"]) # cosine scheduler
    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": 1.3707612106137713e-05, "weight_decay": weight_decay_reg, "rho": 0.1},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": 4.9057851262173356e-05, "weight_decay": weight_decay_embd, "rho": 0.1}])

    #         # optimizer = SophiaG([
    #         #         {"params": self.regression_model.parameters(), "lr": 1e-4, "weight_decay": weight_decay_reg, "rho": 0.1},
    #         #         {"params": self.representation_model.encoder.parameters(),"lr": 1e-4, "weight_decay": weight_decay_embd, "rho": 0.1}])

    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": 1e-5, "weight_decay": weight_decay_reg, "rho": 0.1},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": 5e-6, "weight_decay": weight_decay_embd, "rho": 0.1}])

    #         # optimizer = SophiaG([
    #         #             {"params": self.regression_model.parameters(), "lr": 1e-3, "weight_decay": weight_decay_reg},
    #         #             {"params": self.representation_model.encoder.parameters(),"lr": 5e-4, "weight_decay": weight_decay_embd}])
            
    #         # optimizer = torch.optim.SGD([
    #         #                 {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": weight_decay_reg},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd}])
    #         # optimizer = torch.optim.SGD([
    #         #                 {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": weight_decay_reg, "momentum": 0.9},
    #         #                 {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd,"momentum": 0.9}])
    #         # optimizer_2nd_order = torch.optim.LBFGS(list(self.regression_model.parameters())+list(self.representation_model.encoder.parameters()))

    #         print("optimizers are properly initalized", flush=True)
    #     else: 
    #         for param in self.regression_model.parameters():
    #             param.requires_grad = True 
    #         optimizer = torch.optim.AdamW(self.regression_model.parameters(), lr=lr_reg, weight_decay=weight_decay_reg)
    #     # scheduler = ExponentialLR(optimizer, gamma=0.9)
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5, verbose=True, min_lr=1e-12)
    #     # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=120, T_mult=2)
    #     # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60)

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     best_loss =  float('inf')
    #     running_loss_prev = torch.inf 
    #     # second_order = False 

    #     # patience = params["patience"]
    #     # patience_ref = 0
    #     threshold = 1e-4
    #     # # progress = 0.99 # for SophiaG
    #     # progress = 0.999 # For switch optimizer
    #     # cooldown = params["cooldown"]
    #     # cooldown_counter = cooldown
    #     # shrink_lr = False 
    #     for epoch in range(1,num_epoch+1):

    #         running_loss = 0
    #         for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             # y_pred_batch = self.regressor(self.forward_embedding(indice_pairs_batch))
    #             if "GP" in self.reg_class:
    #                 with gpytorch.settings.num_likelihood_samples(1000):
    #                     optimizer.zero_grad()
    #                     z1,z2 = self.forward_embedding(self.data.x, indice_pairs_batch)
    #                     # if (epoch == 1) and (inner_iter == 0):
    #                     # self.regressor.set_train_data(inputs=[z1,z2], targets=train_y,strict=False)
    #                     # self.regressor.train()
    #                     y_pred_batch = self.regression_model(torch.hstack([z1,z2]))
    #                     loss = -mll(y_pred_batch,train_y_batch.reshape(-1))
    #                     loss.backward()
    #                     # total_norm = 0.0
    #                     # for param in self.representation_model.encoder.parameters():
    #                     #     if param.grad is not None:
    #                     #         param_norm = param.grad.data.norm(2)  # L2 norm
    #                     #         total_norm += param_norm.item() ** 2
    #                     #         print(f'Gradient norm squared (L2^2) for a GNN parameter: {param_norm:.4f}')
    #                     # total_norm = total_norm ** 0.5
    #                     # print(f'Total gradient norm (L2) across all GNN parameters: {total_norm:.4f}')

    #                     # total_norm = 0.0
    #                     # for param in self.regression_model.parameters():
    #                     #     if param.grad is not None:
    #                     #         param_norm = param.grad.data.norm(2)  # L2 norm
    #                     #         total_norm += param_norm.item() ** 2
    #                     #         print(f'Gradient norm squared (L2^2) for a regression parameter: {param_norm:.4f}')
    #                     # total_norm = total_norm ** 0.5
    #                     # print(f'Total gradient norm (L2) across all regression parameters: {total_norm:.4f}')
    #                     optimizer.step()
    #                     # with torch.no_grad():
    #                     #     print(loss.item())
    #                     #     z1,z2 = self.forward_embedding(self.data.x, indice_pairs_batch)
    #                     #     y_pred_batch = self.regression_model(torch.hstack([z1,z2]))
    #                     #     loss = -mll(y_pred_batch,train_y_batch.reshape(-1))
    #                     #     print(loss.item())
    #                     # breakpoint()
    #             else:
    #                 # if not second_order:
    #                 # def closure():
    #                 #     optimizer.zero_grad()
    #                 #     y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
    #                 #     loss = F.huber_loss(y_pred_batch,train_y_batch)
    #                 #     return loss
    #                 optimizer.zero_grad()
    #                 y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
    #                 # loss = F.mse_loss(y_pred_batch,train_y_batch) 
    #                 loss = F.huber_loss(y_pred_batch,train_y_batch)
    #                 # reweight the loss if it's on data that have been seen before
    #                 # loss = F.mse_loss(y_pred_batch,train_y_batch) if batch_index<=3 else F.mse_loss(y_pred_batch,train_y_batch)*self.loss_reweight# regression loss
    #                 # if self.lambda_ewc is not None:
    #                 #         # Add EWC penalty
    #                 #         for name, param in self.representation_model.encoder.named_parameters():
    #                 #             loss += self.lambda_ewc * torch.sum(self.encoder_fisher_information[name]*((param - self.encoder_opt_params[name])**2))
    #                 loss.backward()
    #                 # total_norm = 0.0
    #                 # for param in self.representation_model.encoder.parameters():
    #                 #     if param.grad is not None:
    #                 #         param_norm = param.grad.data.norm(2)  # L2 norm
    #                 #         total_norm += param_norm.item() ** 2
    #                 #         print(f'Gradient norm squared (L2^2) for a GNN parameter: {param_norm:.4f}')
    #                 # total_norm = total_norm ** 0.5
    #                 # print(f'Total gradient norm (L2) across all GNN parameters: {total_norm:.4f}')

    #                 # total_norm = 0.0
    #                 # for param in self.regression_model.parameters():
    #                 #     if param.grad is not None:
    #                 #         param_norm = param.grad.data.norm(2)  # L2 norm
    #                 #         total_norm += param_norm.item() ** 2
    #                 #         print(f'Gradient norm squared (L2^2) for a regression parameter: {param_norm:.4f}')
    #                 # total_norm = total_norm ** 0.5
    #                 # print(f'Total gradient norm (L2) across all regression parameters: {total_norm:.4f}')
    #                 # breakpoint()
    #                 # torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
    #                 # torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #                 # torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0, norm_type="inf")
    #                 torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0, norm_type="inf")
    #                 optimizer.step()
    #                 # loss=optimizer.step(closure)
    #                 # if shrink_lr and (scheduler.get_last_lr()[0]==scheduler.base_lrs[0]):
    #                 #     print("base learning rate is shrinked by a rate of 0.5",flush=True)
    #                 #     scheduler.base_lrs[:] = [lr*0.5 for lr in scheduler.base_lrs]
    #                 #     shrink_lr = False 
    #                 # scheduler.step() # for CosineAnnealingWarmRestarts scheduler
    #                 # scheduler.step(epoch + batch_index / len(data_loader))
    #                 # else:
    #                 #     optimizer_2nd_order.zero_grad()
    #                 #     y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
    #                 #     loss = F.huber_loss(y_pred_batch,train_y_batch)
    #                 #     loss.backward()
    #                 #     optimizer_2nd_order.step()
    #             running_loss += loss
    #             # running_loss += loss.item() if batch_index<=3 else loss.item()/self.loss_reweight
    #             # print("loss: {}".format(loss.item()),flush=True)
    #         running_loss /= len(data_loader)
    #         # if not second_order:
    #         scheduler.step(running_loss) # for ReduceLROnPlateau scheduler
    #         # second_order = (running_loss>running_loss_prev)
    #         # running_loss_prev = running_loss
    #         if running_loss < best_loss*(1-threshold):
    #             best_reg_param = self.regression_model.state_dict()
    #             best_gnn_param = self.representation_model.state_dict()
    #             best_loss = running_loss 
    #         # if not optimizer.switch_to_second_order:
    #         #     optimizer.scheduler.step(running_loss)
    #         #     if optimizer.sgd.param_groups[0]['lr'] < optimizer.threshold_lr:
    #         #         optimizer.switch_to_second_order = True
    #         #         print("Switching to Quasi-Newton Method")
    #         # else: 
    #         #     if (running_loss > running_loss_prev*progress) and (not shrink_lr):
    #         #         if cooldown_counter:
    #         #             cooldown_counter-=1 
    #         #         else:
    #         #             patience_ref += 1
    #         #     if patience_ref >= patience:
    #         #         shrink_lr = True 
    #         #         patience_ref = 0
    #         #         cooldown_counter = cooldown
    #         # else: 
    #         #     if (running_loss > running_loss_prev*progress) and (not optimizer.switch_to_second_order):
    #         #         if cooldown_counter:
    #         #             cooldown_counter-=1 
    #         #         else:
    #         #             patience_ref += 1
    #         #     if patience_ref >= patience:
    #         #         optimizer.switch_to_second_order = True 
    #         running_loss_prev = running_loss
    #         # if epoch%5==1:
    #         print(f'Device: {self.device.index:05d}, Epoch: {epoch:05d}, Loss: {running_loss:.4f}',flush=True)
    #     self.regression_model.load_state_dict(best_reg_param)
    #     self.representation_model.load_state_dict(best_gnn_param)
    #     self.loss_reweight = self.loss_reweight/(self.loss_reweight+1)
    #     # self.loss_reweight = max(scheduler._last_lr[0]/lr_reg,1e-3) # update the loss-reweight parameter 
    #     return
    
    # def train_joint_hyperparameter_search(self, train_y, indice_pairs, weight_decay_reg, weight_decay_embd, trial):
    #     self.representation_model.encoder.load_state_dict(self.encoder_opt_params_dict)
    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True


    #     lr = trial.suggest_float('lr', 1e-8, 1, log=True)
    #     patience = trial.suggest_int("patience", 5, 20)
    #     factor = trial.suggest_float('factor', 1e-1, 1)
    #     optimizer = torch.optim.AdamW(self.representation_model.encoder.parameters(),lr=lr, weight_decay=weight_decay_embd, betas=(0.9, 0.99))
        
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     # data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     data_loader = DataLoader(dataset, batch_size=50, shuffle=True)
    #     best_loss = torch.inf 
    #     for epoch in range(1,101):

    #         running_loss = 0
    #         for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             optimizer.zero_grad()
    #             y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
    #             loss = F.huber_loss(y_pred_batch,train_y_batch)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #             optimizer.step()
    #             running_loss += loss.item() 
    #         running_loss /= len(data_loader)
    #         scheduler.step(running_loss)

    #         # Report intermediate value for pruning
    #         trial.report(running_loss, epoch)

    #         # Prune trial if not promising
    #         if trial.should_prune():
    #             raise optuna.exceptions.TrialPruned()
    #         if running_loss < best_loss*0.999:
    #             best_loss = running_loss 
    #     return best_loss

    # def train_joint_hyperparameter_search(self, train_y, indice_pairs, weight_decay_reg, weight_decay_embd, trial):
    #     self.representation_model.encoder.load_state_dict(self.encoder_opt_params_dict)
    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True


    #     lr = trial.suggest_float('lr', 1e-8, 1, log=True)
    #     patience = trial.suggest_int("patience", 5, 200)
    #     factor = trial.suggest_float('factor', 1e-1, 1)
    #     optimizer = torch.optim.AdamW(self.representation_model.encoder.parameters(),lr=lr, weight_decay=weight_decay_embd, betas=(0.9, 0.99))
        
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    #     indices = torch.Tensor(indice_pairs).long().T.to(self.device)
    #     train_y = train_y.to(self.device)
    #     best_loss = torch.inf 
    #     # for epoch in range(1,1000):
    #     for epoch in range(1,10001):

    #         optimizer.zero_grad()
    #         y_pred = self.regression_model(*self.forward_embedding(self.data.x, indices)) 
    #         loss = F.huber_loss(y_pred,train_y)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #         optimizer.step()
    #         running_loss = loss.item() 
    #         scheduler.step(running_loss)

    #         # Report intermediate value for pruning
    #         trial.report(running_loss, epoch)

    #         # Prune trial if not promising
    #         if trial.should_prune():
    #             raise optuna.exceptions.TrialPruned()
    #         if running_loss < best_loss*0.999:
    #             best_loss = running_loss 
    #     return best_loss
    
    # def train_joint_hyperparameter_search(self, train_y, indice_pairs, weight_decay_reg, weight_decay_embd, trial):
    #     self.representation_model.encoder.load_state_dict(self.encoder_opt_params_dict)
    #     self.regression_model.reset_parameters()
    #     self.regression_model.train() # turn the dropout on
    #     for param in self.regression_model.parameters():
    #         param.requires_grad = True
    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True


    #     lr_reg= trial.suggest_float('lr_reg', 1e-8, 1e-1, log=True)
    #     lr_embd= trial.suggest_float('lr_embd', 1e-8, 1e-1, log=True)
    #     patience = trial.suggest_int("patience", 5, 200)
    #     factor = trial.suggest_float('factor', 1e-1, 1)
    #     # optimizer = torch.optim.AdamW([
    #     #                     {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": weight_decay_reg, "betas": (0.9, 0.999)},
    #     #                     {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
    #     optimizer = torch.optim.AdamW([
    #                         {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": 0, "betas": (0.9, 0.999)}, # for inner product no need for weight decay only single bias parameter
    #                         {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
        
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    #     indices = torch.Tensor(indice_pairs).long().T.to(self.device)
    #     train_y = train_y.to(self.device)
    #     best_loss =  float('inf')
    #     for epoch in range(1,10001):

    #         optimizer.zero_grad()
    #         y_pred = self.regression_model(*self.forward_embedding(self.data.x, indices)) 
    #         loss = F.huber_loss(y_pred,train_y)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #         torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
    #         optimizer.step()
    #         running_loss = loss.item() 
    #         scheduler.step(running_loss)

    #         # Report intermediate value for pruning
    #         trial.report(running_loss, epoch)

    #         # Prune trial if not promising
    #         if trial.should_prune():
    #             raise optuna.exceptions.TrialPruned()
    #         if running_loss < best_loss*0.999:
    #             best_loss = running_loss 
    #     return best_loss

    def train_joint_hyperparameter_search(self, train_y, indice_pairs, weight_decay_reg, weight_decay_embd, trial):
        self.representation_model.encoder.load_state_dict(self.encoder_opt_params_dict)
        self.regression_model.reset_parameters()
        self.regression_model.train() # turn the dropout on
        for param in self.regression_model.parameters():
            param.requires_grad = True
        for param in self.representation_model.encoder.parameters():
            param.requires_grad = True


        lr_reg= trial.suggest_float('lr_reg', 1e-8, 1e-1, log=True)
        lr_embd= trial.suggest_float('lr_embd', 1e-8, 1e-1, log=True)
        patience = trial.suggest_int("patience", 5, 20)
        factor = trial.suggest_float('factor', 1e-1, 1)
        lambda_ewc = trial.suggest_float('lambda_ewc', 1e-10, 100, log=True)
        optimizer = torch.optim.AdamW([
                            {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": 0 if self.reg_class=="Inner_product" else weight_decay_reg, "betas": (0.9, 0.999)}, # for inner product no need for weight decay only single bias parameter
                            {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

        dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
        # data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
        data_loader = DataLoader(dataset, batch_size=100,shuffle=True) #shuffle + increased batch size
        best_loss =  float('inf')
        for epoch in range(1,151):

            running_loss = 0
            for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
                train_y_batch = train_y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
                loss = F.huber_loss(y_pred_batch,train_y_batch)
                if self.lambda_ewc is not None:
                    # Add EWC penalty
                    for name, param in self.representation_model.encoder.named_parameters():
                        loss += lambda_ewc * torch.sum(torch.abs(self.encoder_fisher_information[name])*(param - self.encoder_opt_params[name]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item() 
            running_loss /= len(data_loader)
            scheduler.step(running_loss)

            # Report intermediate value for pruning
            trial.report(running_loss, epoch)

            # Prune trial if not promising
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if running_loss < best_loss*0.999:
                best_loss = running_loss 
        return best_loss

    # def train_joint_hyperparameter_search(self, train_y, indice_pairs, weight_decay_reg, weight_decay_embd, trial):
    #     self.representation_model.encoder.load_state_dict(self.encoder_opt_params_dict)
    #     self.regression_model.reset_parameters()
    #     self.regression_model.train() # turn the dropout on
    #     for param in self.regression_model.parameters():
    #         param.requires_grad = True
    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True


    #     lr_reg= trial.suggest_float('lr_reg', 1e-8, 1e-3, log=True)
    #     lr_embd= trial.suggest_float('lr_embd', 1e-8, 1e-3, log=True)
    #     T0 = trial.suggest_int("T0", 50, 200)
    #     patience = trial.suggest_int("patience", 2, 20)
    #     cooldown = trial.suggest_int("cooldown", 0, 5)
    #     factor = trial.suggest_float("factor", 0, 1)
    #     optimizer = torch.optim.AdamW([
    #                         {"params": self.regression_model.parameters(), "lr": lr_reg, "weight_decay": weight_decay_reg, "betas": (0.9, 0.999)},
    #                         {"params": self.representation_model.encoder.parameters(),"lr": lr_embd, "weight_decay": weight_decay_embd, "betas": (0.9, 0.99)}])
        
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0)

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     best_loss =  float('inf')
    #     running_loss_prev = torch.inf 

    #     # patience = 5
    #     patience_ref = 0
    #     threshold = 1e-4
    #     progress = 0.99
    #     # cooldown = 2
    #     cooldown_counter = cooldown
    #     shrink_lr = False 
    #     for epoch in range(1,101):

    #         running_loss = 0
    #         for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             optimizer.zero_grad()
    #             y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch)) 
    #             loss = F.huber_loss(y_pred_batch,train_y_batch)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.regression_model.parameters(), 1.0)
    #             torch.nn.utils.clip_grad_norm_(self.representation_model.encoder.parameters(), 1.0)
    #             optimizer.step()
    #             scheduler.step()
    #             if shrink_lr and (scheduler.get_last_lr()[0]==scheduler.base_lrs[0]):
    #                 scheduler.base_lrs[:] = [lr*factor for lr in scheduler.base_lrs]
    #                 shrink_lr = False 
    #             running_loss += loss.item() 
    #         running_loss /= len(data_loader)

    #         # Report intermediate value for pruning
    #         trial.report(running_loss, epoch)

    #         # Prune trial if not promising
    #         if trial.should_prune():
    #             raise optuna.exceptions.TrialPruned()
    #         if running_loss < best_loss*(1-threshold):
    #             best_loss = running_loss 
    #         else: 
    #             if (running_loss > running_loss_prev*progress) and (not shrink_lr):
    #                 if cooldown_counter:
    #                     cooldown_counter-=1 
    #                 else:
    #                     patience_ref += 1
    #             if patience_ref >= patience:
    #                 shrink_lr = True 
    #                 patience_ref = 0
    #                 cooldown_counter = cooldown
    #         running_loss_prev = running_loss
    #     return best_loss


    # def train_joint_hyperparameter_search(self, train_y, indice_pairs, trial):
    #     self.representation_model.encoder.load_state_dict(self.encoder_opt_params_dict)
    #     self.regression_model.reset_parameters()
    #     self.regression_model.train() # turn the dropout on
    #     for param in self.regression_model.parameters():
    #         param.requires_grad = True
    #     for param in self.representation_model.encoder.parameters():
    #         param.requires_grad = True
        
    #     lr_1st= trial.suggest_float('lr_1st', 1e-6, 1e-4, log=True)
    #     lr_2nd = trial.suggest_float('lr_2nd', 1e-3, 1, log=True)

    #     patience = trial.suggest_int("patience", 2, 10)
    #     cooldown = trial.suggest_int("cooldown", 0, 5)

    #     optimizer = SwitchableOptimizer (list(self.regression_model.parameters())+list(self.representation_model.encoder.parameters()), lr_2nd=lr_2nd, lr_1st=lr_1st) # cosine scheduler

    #     dataset = TensorDataset(torch.Tensor(indice_pairs).long().T.to(self.device),train_y)
    #     data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    #     best_loss =  float('inf')
    #     running_loss_prev = torch.inf 

    #     patience_ref = 0
    #     threshold = 1e-4
    #     progress = 0.999
    #     cooldown_counter = cooldown
    #     for epoch in range(1,101):

    #         running_loss = 0
    #         for batch_index, (indice_pairs_batch, train_y_batch) in enumerate(data_loader):
    #             train_y_batch = train_y_batch.to(self.device)
    #             def closure():
    #                 optimizer.zero_grad()
    #                 y_pred_batch = self.regression_model(*self.forward_embedding(self.data.x, indice_pairs_batch))
    #                 loss = F.huber_loss(y_pred_batch,train_y_batch)
    #                 return loss
    #             loss_item=optimizer.step(closure)
    #             running_loss += loss_item
    #         running_loss /= len(data_loader)

    #         # Report intermediate value for pruning
    #         trial.report(running_loss, epoch)

    #         # Prune trial if not promising
    #         if trial.should_prune():
    #             raise optuna.exceptions.TrialPruned()
            
    #         if running_loss < best_loss*(1-threshold):
    #             best_loss = running_loss 
    #         else: 
    #             if (running_loss > running_loss_prev*progress) and (not optimizer.switch_to_second_order):
    #                 if cooldown_counter:
    #                     cooldown_counter-=1 
    #                 else:
    #                     patience_ref += 1
    #             if patience_ref >= patience:
    #                 optimizer.switch_to_second_order = True 
    #         running_loss_prev = running_loss
    #     return best_loss
