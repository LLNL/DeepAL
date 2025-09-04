import gc
import pickle
from itertools import combinations, product, combinations_with_replacement

from copy import deepcopy as deepcopy

import numpy as np
from numpy.random import default_rng

import json

import torch
from torch_geometric.data.hetero_data import HeteroData
from methods.active_learning.active_learning_model import ActiveLearningModel

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.pruners import HyperbandPruner

from methods.representation_learning.Active_RGCN import ActiveGNN
from methods.representation_learning.Active_FE import ActiveFE


class Active_Representation(ActiveLearningModel):
    """
        Active Representation Learning based approach to update the embedding
        while perform arm recommendations
    """
    def __init__(self, model_config=None, name="Active_Representation", device=None, **kwargs):
        """
            model_config: path to file containing the features or embeddings for each point
        """ 
        super().__init__(model_config, name)
        if model_config is None:
            raise ValueError("A config file is required for this model")
        with open(model_config) as f:
            model_config = json.load(f)

        # config for training
        self.training_config = model_config['training']
        # config for representation and regression models
        self.representation_model_config = model_config['model']["representation_model"]
        self.regression_model_config = model_config['model']["regression_model"]
        self.memory_efficient = model_config['model']["memory_efficient"]
        self.filter_mask = None
        
        if device is None:
            # set device
            if (not torch.cuda.is_available() or not torch.cuda.device_count()) and (self.training_config['device'] != 'cpu'):
                print ("CUDA is not available. Running on CPU.")
                self.training_config['device'] = 'cpu'
                self.device = torch.device('cpu')
            elif self.training_config['device'].startswith('cuda'):
                self.device = torch.device(self.training_config['device'])
            elif self.training_config['device'] == 'gpu':
                self.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1)) #get the last device available
                # self.device = torch.device('cuda:0') #get the first device available
            elif self.training_config['device'] == 'cpu':
                self.device = torch.device('cpu')
            else:
                raise ValueError("Unexpected device name: %s" % self.training_config['device'])
        else:
            self.device = device 

        # TODO:
        #     1. Refine the solution here to be compatiable with other models
        #     2. Change the names so it's not limited to only the HIV dataset
        self.hiv_indices = None
        if self.representation_model_config['model_class'] == "ActiveGNN":
            self.graph_file = model_config['data_file']
            self.model = self._make_model().to(self.device)

            # dirty solution to made compatible with the new free embedding (i.e., random initialized embeddings but trainingable) 
            #  

            with torch.no_grad():
                self.nodes_embedding = self.model.representation_model.encode(self.model.data.x, 
                                                                            self.model.data.edge_index, 
                                                                            self.model.data.edge_type).detach().clone()
            if self.hiv_indices is not None:
                self.nodes_embedding = self.nodes_embedding[self.hiv_indices]
            self.edges_clustering = None # clusters of the edges  based on embedding of the data  (dictionary that maps each indices -> cluster)
            self.edges_clusters = None # clusters of the edges  based on embedding of the data  (dictionary that maps each cluster -> list of indices)
            self.nodes_clusters = None # clusters of the nodes based on embedding of the data (dictionary that maps each cluster -> list of indices)        
        elif self.representation_model_config['model_class'] == "ActiveFE":
            self.model = self._make_model().to(self.device)
            with torch.no_grad():
                self.nodes_embedding = self.model.representation_model().detach().clone()
    
    def init_model(self):
        """
            Initialize the representation model using the encoder-decoder scheme 
        """
        self.model.init_train()
        return 
    
    def re_init(self):
        """
            Re-initialize the representation model
        """
        print("Re-initialize the model.....", flush=True)
        # self.model.representation_model.encoder.load_state_dict(self.model.encoder_opt_params_dict)
        self.model.regression_model.reset_parameters()
        return  
    
    def hyper_para_selection(self, all_targets, seen_mask):
        indices = np.arange(len(seen_mask))[seen_mask] # find indices of points that have been seen
        first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices))
        indice_pairs=[list(first_indices), list(second_indices)]
        train_y = all_targets[seen_mask].view(-1, 1)

        # Create study with Hyperband pruner
        # pruner = HyperbandPruner(min_resource=2000)
        pruner = HyperbandPruner(min_resource=20) # for stochastic
        # pruner = HyperbandPruner(min_resource=100)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_switch_100iters.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_adamw_cos_100iters.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_inner_adamw_decay_100iters.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_inner_adamw_decay_100iters_updated.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_inner_adamw_decay_100iters_updated_random.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_inner_adamw_nonstochastic_10000iters.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_adamw_cos.db", load_if_exists=True)
        # study = optuna.create_study(study_name="tuning the optimizer", direction='minimize', pruner=pruner, storage="sqlite:///trials_lassen_adamw_decay.db", load_if_exists=True)


        # Create an RDBStorage object using the custom engine's URL
        # storage = optuna.storages.RDBStorage(url="sqlite:///trials_lassen_inner_adamw_nonstochastic_10000iters.db", engine_kwargs={"connect_args": {"timeout": 100}})
        # storage = JournalStorage(JournalFileStorage("trials_lassen_inner_adamw_nonstochastic_10000iters.log"))
        # storage = JournalStorage(JournalFileStorage("trials_lassen_inner_adamw_150epochs_wbias.log")) #0.2 dropout
        # storage = JournalStorage(JournalFileStorage("trials_lassen_inner_adamw_150epochs_wbias_wtdropout.log")) 
        # storage = JournalStorage(JournalFileStorage("trials_lassen_bilinear_adamw_nonstochastic_10000iters.log"))
        # storage = JournalStorage(JournalFileStorage("trials_lassen_diagonal_adamw_nonstochastic_10000iters.log"))
        storage = JournalStorage(JournalFileStorage("trials_lassen_bilinear_adamw_150epochs_ewc_wtdropout_corrected.log")) 

        # Load the study using the custom storage
        study = optuna.create_study(study_name="tuning the optimizer",  direction='minimize', pruner=pruner, storage=storage, load_if_exists=True)

        # Define a wrapper function to pass additional parameters
        def wrapped_objective(trial):
            return self.model.train_joint_hyperparameter_search(train_y, indice_pairs, self.training_config["weight_decay_reg"], self.training_config["weight_decay_embd"], trial) #tuning for sophiaG/Lion/AdamW
            # return self.model.train_joint_hyperparameter_search(train_y, indice_pairs, trial) # tuning for switch-optimizer
        study.optimize(wrapped_objective, n_trials=40)

        # Print the best hyperparameters
        print("Best hyperparameters: ", study.best_trial.params, flush=True)
        print("Best value: ", study.best_value, flush=True)
        return 
    
    def fit(self, all_targets, seen_mask, ensemble_size=1):
        """
            Fitting the active representation model
        """
        # NOTE TO SELF: num_epoch_embedding controls whether model will be trained jointly or not
        # data preparation
        # get the training data for points that have been already seen
        if self.representation_model_config['model_class'] == "ActiveGNN" and self.representation_model_config["params"]["lambda_ewc"] == "auto":
            self.model.lambda_ewc = 200
        indices = np.arange(len(seen_mask))[seen_mask] # find indices of points that have been seen
        first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices))
        indice_pairs=[list(first_indices), list(second_indices)]
        train_y = all_targets[seen_mask].view(-1, 1)

        if self.training_config['train_jointly']:
            if self.model.update_embedding:
                self.model.train_joint_non_alternating(train_y, indice_pairs, 
                                                        num_epoch=self.training_config['num_epoch_joint'],
                                                        # num_epoch=20,
                                                        lr_reg = self.training_config["learning_rate_regression"],
                                                        lr_embd = self.training_config["learning_rate_embedding"],
                                                        weight_decay_reg = self.training_config["weight_decay_reg"],
                                                        weight_decay_embd = self.training_config["weight_decay_embd"],
                                                        gamma = self.training_config["learning_rate_decay"],
                                                        ensemble_size=ensemble_size
                                                        )
            else:
                self.model.train_joint_non_alternating(train_y, indice_pairs, 
                                                        num_epoch=self.training_config['num_epoch_joint'],
                                                        # num_epoch=20,
                                                        lr_reg = self.training_config["learning_rate_regression"],
                                                        lr_embd = self.training_config["learning_rate_embedding"],
                                                        weight_decay_reg = self.training_config["weight_decay_reg"],
                                                        weight_decay_embd = self.training_config["weight_decay_embd"],
                                                        gamma = self.training_config["learning_rate_decay"],
                                                        embeddings = self.nodes_embedding,
                                                        ensemble_size=ensemble_size
                                                        )
        else: # train alternating between embedding model and regressor
            for _ in range(self.training_config['num_epoch_alternating']):
                self.model.train_joint(train_y, indice_pairs, 
                                        num_epoch=self.training_config['num_epoch_regressor'],
                                        num_epoch_embedding=self.training_config['num_epoch_embedding'],
                                        lr_reg = self.training_config["learning_rate_regression"],
                                        lr_embd = self.training_config["learning_rate_embedding"],
                                        weight_decay_reg = self.training_config["weight_decay_reg"],
                                        weight_decay_embd = self.training_config["weight_decay_embd"],
                                        gamma = self.training_config["learning_rate_decay"]
                                       )
        if self.representation_model_config['model_class'] == "ActiveGNN" and self.model.update_embedding:
            with torch.no_grad():
                self.nodes_embedding = self.model.representation_model.encode(self.model.data.x, 
                                                                            self.model.data.edge_index, 
                                                                            self.model.data.edge_type).detach().clone()
            if self.hiv_indices is not None:
                self.nodes_embedding = self.nodes_embedding[self.hiv_indices]

        elif self.representation_model_config['model_class'] == "ActiveFE":
            with torch.no_grad():
                self.nodes_embedding = self.model.representation_model().detach().clone()
        return

    def update(self, all_targets, seen_mask, new_samples, early_stop=False):
        """
            Updating the active representation model given new samples
        """

        # data preparation
        # get the training data for points that have been already seen
        indices = np.arange(len(seen_mask))[seen_mask] # find indices of points that have been seen
        if self.representation_model_config['model_class'] == "ActiveGNN" and self.representation_model_config["params"]["lambda_ewc"]=="auto":
            self.model.lambda_ewc /= np.e # update the penalty terms
        if self.training_config['train_jointly']:
            indices = np.concatenate([new_samples, indices]) # ensure new samples are included in the data
            first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices))
            indice_pairs=[list(first_indices), list(second_indices)]
            train_y = all_targets[[indices]].view(-1,1)
            if self.model.update_embedding:
                self.model.train_joint_non_alternating(train_y, indice_pairs, 
                                                        num_epoch=20 if early_stop else self.training_config['num_epoch_joint'],
                                                        lr_reg = self.training_config["learning_rate_regression"],
                                                        lr_embd = self.training_config["learning_rate_embedding"], 
                                                        # lr_reg = self.training_config["learning_rate_regression"] if early_stop else self.training_config["learning_rate_regression"]*0.01,
                                                        # lr_embd = self.training_config["learning_rate_embedding"] if early_stop else self.training_config["learning_rate_embedding"]*0.1,
                                                        weight_decay_reg = self.training_config["weight_decay_reg"],
                                                        weight_decay_embd = self.training_config["weight_decay_embd"],
                                                        gamma = self.training_config["learning_rate_decay"]
                                                        )
            else:
                self.model.train_joint_non_alternating(train_y, indice_pairs, 
                                        num_epoch=20 if early_stop else self.training_config['num_epoch_joint'],
                                        lr_reg = self.training_config["learning_rate_regression"],
                                        lr_embd = self.training_config["learning_rate_embedding"], 
                                        # lr_reg = self.training_config["learning_rate_regression"] if early_stop else self.training_config["learning_rate_regression"]*0.01,
                                        # lr_embd = self.training_config["learning_rate_embedding"] if early_stop else self.training_config["learning_rate_embedding"]*0.1,
                                        weight_decay_reg = self.training_config["weight_decay_reg"],
                                        weight_decay_embd = self.training_config["weight_decay_embd"],
                                        gamma = self.training_config["learning_rate_decay"],
                                        embeddings = self.nodes_embedding
                                        )
        else: # train alternating between embedding model and regressor
            for _ in range(self.training_config['num_epoch_alternating']):
                indices = np.concatenate([indices,new_samples]) # ensure new samples are included in the data
                first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices))
                indice_pairs=[list(first_indices), list(second_indices)]
                train_y = all_targets[[indices]].view(-1,1)
                self.model.train_joint(train_y, indice_pairs, 
                                        num_epoch=self.training_config['num_epoch_regressor'],
                                        num_epoch_embedding=self.training_config['num_epoch_embedding'],
                                        lr_reg = self.training_config["learning_rate_regression"],
                                        lr_embd = self.training_config["learning_rate_embedding"],
                                        weight_decay_reg = self.training_config["weight_decay_reg"],
                                        weight_decay_embd = self.training_config["weight_decay_embd"],
                                        gamma = self.training_config["learning_rate_decay"]
                                       )
        if self.representation_model_config['model_class'] == "ActiveGNN" and self.model.update_embedding:
            with torch.no_grad():
                self.nodes_embedding = self.model.representation_model.encode(self.model.data.x, self.model.data.edge_index, self.model.data.edge_type).detach().clone()
            if self.hiv_indices is not None:
                self.nodes_embedding = self.nodes_embedding[self.hiv_indices]
        elif self.representation_model_config['model_class'] == "ActiveFE":
            with torch.no_grad():
                self.nodes_embedding = self.model.representation_model().detach().clone()
        return


    def _predict(self, indices, acquisition_fn='eps-greedy', max_test=600, n_rand=100, return_grad_norm=False):
        
        # use yield statement to have more memory efficient code
        cur_ind = 0
        while cur_ind < len(indices):

            # prepare input to deep model
            cur_indices = indices[cur_ind:(cur_ind + max_test)]
            first_indices, second_indices = zip(*map(lambda key: self.index_map[key], cur_indices))

            with torch.no_grad():
                z1 = self.nodes_embedding[first_indices,:].detach().clone()
                z2 = self.nodes_embedding[second_indices,:].detach().clone()
                if "variance" in acquisition_fn:
                    if "GP" in self.model.reg_class:
                        posterior_mean, posterior_var = self.model.predict([z1,z2],return_var=True)
                        if acquisition_fn == "variance":
                            yield posterior_var.detach().reshape(-1)
                        else:
                            yield posterior_mean.detach().reshape(-1), posterior_var.detach().reshape(-1)
                    else:
                        # use dropout to enable variance estimation
                        # not peroforming well
                        if self.model.diagonal or self.model.non_regression:
                            posterior = self.model.predict((z1,z2),acquisition_fn=acquisition_fn, if_dropout=True).detach().reshape(-1)
                        else:
                            posterior_mean = self.model.predict((z1,z2),acquisition_fn=acquisition_fn, if_dropout=True).detach().reshape(-1)
                            posterior_var = np.zeros(posterior_mean.shape)
                            for niter in range(1,n_rand):
                                posterior_new = self.model.predict((z1,z2),acquisition_fn=acquisition_fn, if_dropout=True).detach().numpy().reshape(-1)
                                posterior_prev = posterior_mean.copy()
                                posterior_mean = niter/(niter+1)*posterior_mean+posterior_new/(niter+1)
                                posterior_var =  niter/(niter+1)*posterior_var+(posterior_new-posterior_mean)*(posterior_new-posterior_prev)/(niter+1)
                        yield posterior_var
                else:
                    if return_grad_norm:
                        posterior = self.model.predict((z1,z2),acquisition_fn=acquisition_fn).detach().reshape(-1)
                        grad_norm = torch.linalg.vector_norm(z1*z2,axis=1)
                        yield posterior,grad_norm
                    else:
                        if "GP" in self.model.reg_class:
                            posterior = self.model.predict([z1,z2],acquisition_fn=acquisition_fn).detach().reshape(-1)
                        else:
                            posterior = self.model.predict((z1,z2),acquisition_fn=acquisition_fn).detach().reshape(-1)
                        yield posterior
                cur_ind += max_test

    def predict(self, indices, acquisition_fn='eps-greedy', n_rand=100, return_grad_norm=False,return_embedding=False):

        # gc.collect()
        # torch.cuda.empty_cache()

        # prepare input to deep model
        first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices))
        with torch.no_grad():
            z1 = self.nodes_embedding[list(first_indices),:].detach().clone()
            z2 = self.nodes_embedding[list(second_indices),:].detach().clone()
            if "variance" in acquisition_fn:
                if "GP" in self.model.reg_class:
                    posterior_mean, posterior_var = self.model.predict((z1,z2),return_var=True)
                    if acquisition_fn == "variance":
                        return posterior_var.detach().reshape(-1)
                    else: 
                        return posterior_mean.detach().reshape(-1), posterior_var.detach().reshape(-1)
                else:
                    # use dropout to enable variance estimation
                    # not peroforming well
                    posterior_mean = self.model.predict((z1,z2),acquisition_fn=acquisition_fn, if_dropout=True).detach().numpy().reshape(-1)
                    posterior_var = np.zeros(posterior_mean.shape)
                    for niter in range(1,n_rand):
                        posterior_new = self.model.predict((z1,z2),acquisition_fn=acquisition_fn, if_dropout=True).detach().numpy().reshape(-1)
                        posterior_prev = posterior_mean.copy()
                        posterior_mean = niter/(niter+1)*posterior_mean+posterior_new/(niter+1)
                        posterior_var =  niter/(niter+1)*posterior_var+(posterior_new-posterior_mean)*(posterior_new-posterior_prev)/(niter+1)
                    return posterior_var
            else:
                posterior = self.model.predict((z1,z2),acquisition_fn=acquisition_fn).detach().reshape(-1)
                if return_grad_norm:
                    grad_norm = torch.linalg.vector_norm(z1*z2,axis=1)
                    return posterior,grad_norm
                elif return_embedding:
                    return posterior,(z1,z2)
                else:
                    return posterior
                
    def acquisition_diversification(self, n_samples, seen_mask, n_round, rng=default_rng(1234)):
        return 
    
    def acquisition_max_optimism(n_samples, seen_mask, n_round):
        return 
    
    def acquisition_eps_greedy(self, n_samples, seen_mask, n_round, rng=default_rng(1234)):
        """
            epsilon greedy, choose (1-eps) of the samples greedily while the other
            (eps) of the samples randomly
        """

        # find indices of points that have not been seen yet
        if self.filter_mask is None:
            indices = np.arange(len(seen_mask))[(~seen_mask)]
        else:
            print("applying pre-filtering", flush=True)
            indices = np.arange(len(seen_mask))[(~seen_mask)*self.filter_mask]
        n_rand = int(np.round(self.representation_model_config["params"]["eps"]*n_samples)) # numer of of random samples
        # n_rand = int(np.round((1-n_round)*n_samples)) # numer of of random samples
        n_greedy =  n_samples-n_rand # number of greedy samples
        print("number of greedy samples: {}, and number of random samples: {}".format(n_greedy, n_rand), flush=True)

        if self.memory_efficient:
            y_eps_greedy = torch.empty(0)
            for posterior in self._predict(indices):
                y_eps_greedy = torch.cat([y_eps_greedy,posterior.cpu()])
        else:
            y_eps_greedy = self.predict(indices).cpu()
        
        new_samples_greedy = indices[torch.argsort(y_eps_greedy)][:n_greedy]
        indices_rand = np.setdiff1d(indices,new_samples_greedy,assume_unique=True)
        # new_samples_greedy_rand = rng.choice(indices_rand, size=n_rand, replace=False)
        new_samples_greedy_rand = np.random.choice(indices_rand, size=n_rand, replace=False)
        new_samples = np.concatenate([new_samples_greedy, new_samples_greedy_rand])


        return new_samples

    def acquisition_greedy_div(self, n_samples, seen_mask, n_round=0):
        """
            greedy_div, a variant of epsilone greedy acquisition strategy
            that prioritize diverse set of arms based on clustering on the
            current embedded subspace
        """

        # n_greedy = int(n_round*n_samples) # numer of of greedy samples
        n_greedy = n_samples # numer of of greedy samples
        # n_greedy = n_samples-int(np.round(self.model_params["eps"]*n_samples))

        # find indices of points that have not been seen yet
        indices = np.arange(len(seen_mask))[~seen_mask]

        samples_per_cluster = n_greedy//len(self.edges_clusters)
        remainder_samples=n_greedy%len(self.edges_clusters)
        new_samples = None
        for idx, key in enumerate(self.edges_clusters.keys()):
            indices_per_cluster = np.intersect1d(indices,self.edges_clusters[key])
            y_greedy = np.empty(0)
            for posterior in self._predict(indices_per_cluster):
                y_greedy =np.concatenate([y_greedy,posterior])
            if remainder_samples>idx:
                if len(y_greedy)< samples_per_cluster+1:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)]
                else:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)][:samples_per_cluster+1]
            else:
                if len(y_greedy)< samples_per_cluster:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)]
                else:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)][:samples_per_cluster]
            if new_samples is None:
                new_samples = new_samples_per_cluster
            else:
                new_samples = np.concatenate([new_samples, new_samples_per_cluster])

        # dirty solution if there wasn't enough samples
        # selected from the cluster just random sample
        if n_samples>len(new_samples):
            new_indices = np.setdiff1d(indices,new_samples,assume_unique=True)
            new_samples = np.concatenate([new_samples,np.random.choice(new_indices, size=n_samples-len(new_samples), replace=False)])
        return new_samples

    def acquisition_badge(self, n_samples, seen_mask, n_round=0):
        """
            greedy_explore, a variant of greedy acquisition strategy
            that explore eps portion of arms based on gradient information of 
            the parameters 
        """

        # find indices of points that have not been seen yet
        if self.filter_mask is None:
            indices = np.arange(len(seen_mask))[~seen_mask]
        else:
            indices = np.arange(len(seen_mask))[(~seen_mask)*self.filter_mask]

        if self.memory_efficient:
            y_greedy = torch.empty(0)
            grad_norms = torch.empty(0)
            for posterior, grad_norm in self._predict(indices, return_grad_norm=True):
                y_greedy = torch.cat([y_greedy,posterior.cpu()])
                grad_norms = torch.cat([grad_norms,grad_norm.cpu()])
        else:
            y_greedy, grad_norms = self.predict(indices, return_grad_norm=True)

        new_samples = indices[torch.argsort(grad_norms.cpu())][-n_samples:]

        return new_samples

    def acquisition_greedy_explore(self, n_samples, seen_mask, n_round=0):
        """
            greedy_explore, a variant of greedy acquisition strategy
            that explore eps portion of arms based on gradient information of 
            the parameters 
        """

        # n_greedy = int(n_round*n_samples) # numer of of greedy samples
        # n_greedy = n_samples # numer of of greedy samples
        if n_round == 0:
            # find indices of points that have not been seen yet
            if self.filter_mask is None:
                indices = np.arange(len(seen_mask))[~seen_mask]
            else:
                indices = np.arange(len(seen_mask))[(~seen_mask)*self.filter_mask]

            if self.memory_efficient:
                y_greedy = torch.empty(0)
                grad_norms = torch.empty(0)
                for posterior, grad_norm in self._predict(indices, return_grad_norm=True):
                    y_greedy = torch.cat([y_greedy,posterior.cpu()])
                    grad_norms = torch.cat([grad_norms,grad_norm.cpu()])
            else:
                y_greedy, grad_norms = self.predict(indices, return_grad_norm=True)

            new_samples = indices[torch.argsort(grad_norms.cpu())][-n_samples:]

        else:
            n_greedy = int(np.round(n_round*n_samples))
            # n_greedy = 0
            # n_explore = int(np.round(self.model_params["eps"]*n_samples))
            # n_greedy = n_samples-n_explore
            n_explore= n_samples-n_greedy

            # find indices of points that have not been seen yet
            if self.filter_mask is None:
                indices = np.arange(len(seen_mask))[~seen_mask]
            else:
                indices = np.arange(len(seen_mask))[(~seen_mask)*self.filter_mask]

            if self.memory_efficient:
                y_greedy = torch.empty(0)
                grad_norms = torch.empty(0)
                for posterior, grad_norm in self._predict(indices, return_grad_norm=True):
                    y_greedy = torch.cat([y_greedy,posterior.cpu()])
                    grad_norms = torch.cat([grad_norms,grad_norm.cpu()])
            else:
                y_greedy, grad_norms = self.predict(indices, return_grad_norm=True)

            new_samples_greedy = indices[torch.argsort(y_greedy).cpu()][:n_greedy]
            indices_explore = np.setdiff1d(indices,new_samples_greedy,assume_unique=True)
            grad_norms = grad_norms[np.array([np.where(indice==indices)[0][0] for indice in indices_explore])]
            new_samples_greedy_explore = indices_explore[torch.argsort(grad_norms.cpu())][-n_explore:]
            new_samples = np.concatenate([new_samples_greedy, new_samples_greedy_explore])

        return new_samples
    
    def acquisition_bait(self, n_samples, seen_mask, n_round=0):
        """
            bait based on following paper: 
                Ash, Jordan, et al. "Gone fishing: Neural active learning with fisher embeddings." 
                Advances in Neural Information Processing Systems 34 (2021): 8927-8939.
                link: https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf
        """
        lamb = 0.01 #stablize inversion of fisher matrix 
        rank = 1 # the rank for the low rank update
                # note: since we are predicting a scalar target variable it will be set to 1 

        n_greedy = int(n_round*n_samples) # numer of of greedy samples
        # n_greedy = n_samples # numer of of greedy samples
        # n_greedy = 0
        n_explore= n_samples-n_greedy

        # find indices of points that have not been seen yet
        indices = np.arange(len(seen_mask))
        indices_unseen = np.arange(len(seen_mask))[~seen_mask]
        indices_seen = np.arange(len(seen_mask))[seen_mask]
        n_seen = len(indices_seen)
        with torch.no_grad():
            if n_explore>0:
                first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices_seen))
                z1 = deepcopy(self.nodes_embedding[first_indices,:].detach())
                z2 = deepcopy(self.nodes_embedding[second_indices,:].detach())
                (n,d) = z1.shape
                # embeddings = deepcopy(torch.cat((z1,z2,torch.einsum('ij,ik->ijk', z1, z2).reshape(n, d**2)+torch.einsum('ij,ik->ijk', z2, z1).reshape(n, d**2)),dim=1).detach())
                embeddings = deepcopy((torch.einsum('ij,ik->ijk', z1, z2).reshape(n, d**2)+torch.einsum('ij,ik->ijk', z2, z1).reshape(n, d**2)).detach())

                fisher_seen = (embeddings.T@embeddings).detach().to('cuda:0') #torch.einsum('ki,kj->ij', embeddings,embeddings) 
            first_indices, second_indices = zip(*map(lambda key: self.index_map[key], indices_unseen))
            z1 = deepcopy(self.nodes_embedding[first_indices,:].detach())
            z2 = deepcopy(self.nodes_embedding[second_indices,:].detach())
            (n,d) = z1.shape
            mask = np.ones(n, dtype=bool)
            # dim = d*(d+2)
            dim = d**2
            # embeddings = deepcopy(torch.cat((z1,z2,torch.einsum('ij,ik->ijk', z1, z2).reshape(n, d**2)+torch.einsum('ij,ik->ijk', z2, z1).reshape(n, d**2)),dim=1).detach())
            embeddings = deepcopy((torch.einsum('ij,ik->ijk', z1, z2).reshape(n, d**2)+torch.einsum('ij,ik->ijk', z2, z1).reshape(n, d**2)).detach())
            y_greedy = deepcopy(self.model.predict(embeddings).detach()).reshape(-1)
            embeddings=embeddings.to('cuda:0')
            if n_explore>0:
                fisher_global = (embeddings.T@embeddings+fisher_seen)/len(seen_mask)
                fisher_seen = fisher_seen/n_seen
                # currentInv = torch.inverse(lamb*torch.eye(dim)+fisher_seen*(n_seen/(n_seen+n_explore)))
                currentInv = torch.inverse(lamb*torch.eye(dim).to('cuda:0')+fisher_seen*(n_seen/(n_seen+n_explore)))
                embeddings = embeddings*np.sqrt(n_explore/(n_explore+n_seen)) # scale the unseen data embeddings
                # forward selection, over-sample by 2x
                print('forward selection...', flush=True)
                over_sample = 2
                indsAll = []

                for i in range(int(over_sample *n_explore)):

                    # check trace with low-rank updates (woodbury identity)
                    # innerInv = 1/(rank + torch.einsum("ij,ij,jj->i", embeddings, embeddings, currentInv)).detach() outdated
                    innerInv = 1/(rank+torch.einsum('ij,ij->i', embeddings, embeddings@currentInv)).detach()
                    if torch.isinf(innerInv).any():
                        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
                    # traceEst = (embeddings @ (currentInv @ (fisher_global @ (currentInv @ (embeddings.T @ innerInv)))))
                    # test = torch.diagonal(embeddings @ (currentInv @ (fisher_global @ (currentInv @ (embeddings.T @ torch.diag(innerInv))))))
                    # test2 = torch.diagonal(embeddings @ (currentInv @ (fisher_global @ (currentInv @ (embeddings.T*innerInv)))))
                    # test3 = torch.diagonal(embeddings @ (currentInv @ fisher_global @ currentInv) @ (embeddings.T*innerInv))
                    traceEst = torch.einsum("ij,ji->i",embeddings, (currentInv @ fisher_global @ currentInv)@(embeddings.T))*innerInv
                    # traceEst = torch.einsum("ij,ij->j",currentInv@(embeddings.T), (fisher_global@currentInv)@embeddings.T)*innerInv
                    # clear out gpu memory
                    # xt = xt_.cpu()
                    # del xt, innerInv
                    # torch.cuda.empty_cache()
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # gc.collect()

                    # get the smallest unselected item
                    # traceEst = traceEst.detach().numpy()
                    # traceEst = traceEst.detach().numpy()
                    traceEst = traceEst.detach().cpu().numpy()
                    for j in np.argsort(traceEst)[::-1]:
                        if j not in indsAll:
                            ind = j
                            break

                    indsAll.append(ind)
                    # print(i, ind, traceEst[ind], flush=True)
                
                    # commit to a low-rank update
                    xt = embeddings[ind,:]
                    innerInv = 1/(rank + xt.dot(currentInv@xt)).detach() #1/(rank + torch.einsum("ij,ij,jj->i", xt, xt, currentInv)).detach()
                    currentInv = (currentInv - torch.outer(currentInv @ (xt*innerInv), xt @ currentInv)).detach()
                # backward pruning
                print('backward pruning...', flush=True)
                for i in range(len(indsAll) - n_explore):

                    # select index for removal
                    xt = embeddings[indsAll,:]
                    innerInv = 1/(-rank + torch.einsum("ij,ij->i", xt,xt@currentInv)).detach()
                    traceEst = torch.einsum("ij,ji->i", xt, currentInv @ (fisher_global @ (currentInv@(xt.T))))*innerInv
                    delInd = torch.argmin(traceEst).item()
                    # print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


                    # low-rank update (woodbury identity)
                    xt = embeddings[indsAll[delInd],:]
                    innerInv = 1/(-rank + xt.dot(currentInv@xt)).detach()
                    currentInv = (currentInv + torch.outer(currentInv @ (xt*innerInv), xt @ currentInv)).detach()
                    del indsAll[delInd]

                del xt, innerInv, currentInv, embeddings,fisher_global,fisher_seen
                # del xt, innerInv, currentInv
                # torch.cuda.empty_cache()
                gc.collect()
                new_samples_explore=indices_unseen[indsAll]
                mask[indsAll]=False
                y_greedy = y_greedy[mask]
            else: 
                new_samples_explore = []
        new_samples_greedy = indices_unseen[mask][torch.argsort(y_greedy).cpu()][:n_greedy]
        # indices_explore = np.setdiff1d(indices_unseen,new_samples_greedy,assume_unique=True)


        new_samples = np.concatenate([new_samples_greedy, new_samples_explore])

        return new_samples
    
    def acquisition_optimism(self):
        # a dumy acquisition strategy for the ensemble model
        return 

    def acquisition_greedy_elim_div(self, n_samples, seen_mask, n_round=0):
        """
            greedy_div, a variant of epsilone greedy acquisition strategy
            that prioritize diverse set of arms based on clustering on the
            current embedded subspace with an elimination scheme
        """
        # n_greedy = int(n_round*n_samples) # numer of of greedy samples
        n_greedy = n_samples # numer of of greedy samples
        # n_greedy = n_samples-int(np.round(self.model_params["eps"]*n_samples))

        # find indices of points that have not been seen yet
        indices = np.arange(len(seen_mask))[~seen_mask]

        # add elimination scheme
                # Get all the confidence band predictions
        y_min_max = np.Inf # Take the worst of each cluster and take the best
        y_per_cluster = dict() # save the greedy predictions per each cluster
        y_best_per_cluster = dict() # save the greedy predictions per each cluster
        keys = list(self.edges_clusters.keys())
        for _, key in enumerate(keys):
            indices_per_cluster = np.intersect1d(indices,self.edges_clusters[key])

            # remove the cluster if we already have
            # explored all the elements
            if len(indices_per_cluster)<1:
                self.clusters.pop(key)
                continue
            y_greedy = np.empty(0)
            for posterior in self._predict(indices_per_cluster):
                y_greedy =np.concatenate([y_greedy,posterior])
            y_min_max = np.minimum(y_min_max, np.max(y_greedy))
            y_per_cluster[key]=y_greedy
            y_best_per_cluster[key] = np.min(y_greedy)

        # Elimination step:
        #   If the best within the cluster with upper confidence band
        #   is not as good as the worst within the optimal cluster then
        #   we will eliminate this cluster from future exploring
        for key,value in y_best_per_cluster.items():
            if value>y_min_max:
                self.edges_clusters.pop(key)

        samples_per_cluster = n_greedy//len(self.edges_clusters)
        remainder_samples=n_greedy%len(self.edges_clusters)
        new_samples = None
        for idx, key in enumerate(self.edges_clusters.keys()):
            indices_per_cluster = np.intersect1d(indices,self.edges_clusters[key])
            y_greedy= y_per_cluster[key]
            if remainder_samples>idx:
                if len(y_greedy)< samples_per_cluster+1:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)]
                else:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)][:samples_per_cluster+1]
            else:
                if len(y_greedy)< samples_per_cluster:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)]
                else:
                    new_samples_per_cluster = indices_per_cluster[np.argsort(y_greedy)][:samples_per_cluster]
            if new_samples is None:
                new_samples = new_samples_per_cluster
            else:
                new_samples = np.concatenate([new_samples, new_samples_per_cluster])

        # dirty solution if there wasn't enough samples
        # selected from the cluster just random sample
        if n_samples>len(new_samples):
            new_indices = np.setdiff1d(indices,new_samples,assume_unique=True)
            new_samples = np.concatenate([new_samples,np.random.choice(new_indices, size=n_samples-len(new_samples), replace=False)])
        return new_samples
    
    def acquisition_annealing(self, n_samples, seen_mask, n_round, rng=default_rng(1234)):
        return 

    def acquisition_var(self, n_samples, seen_mask, n_round, rng=default_rng(1234)):
        """
            maximum variance based strategy
        """

        # find indices of points that have not been seen yet
        # indices = np.arange(len(seen_mask))[~seen_mask]

        # if self.memory_efficient:
        #     y_var = torch.empty(0)
        #     for posterior in self._predict(indices,acquisition_fn="variance"):
        #         y_var = torch.cat([y_var,posterior.cpu()])
        # else:
        #     y_var = self.predict(indices,acquisition_fn="variance").cpu()
        # new_samples = indices[torch.argsort(y_var)][-n_samples:]
        # return new_samples
        return 
    
    def acquisition_variance_mixed(self, n_samples, seen_mask, n_round=0):
        """
            variance_mixed, a variant of variance acquisition strategy
            that explore eps portion of arms based on greedy predictions 
        """

        n_greedy = int(np.round(n_round*n_samples))
        n_var= n_samples-n_greedy

        # find indices of points that have not been seen yet
        indices = np.arange(len(seen_mask))[~seen_mask]

        if self.memory_efficient:
            y_greedy = torch.empty(0)
            y_var = torch.empty(0)
            for posterior, variance in self._predict(indices,acquisition_fn="variance_mixed"):
                y_greedy = torch.cat([y_greedy,posterior.cpu()])
                y_var = torch.cat([y_var,variance.cpu()])
        else:
            y_greedy, y_var = self.predict(indices,acquisition_fn="variance_mixed")

        new_samples_var = indices[torch.argsort(y_var).cpu()][-n_var:]
        indices_greedy = np.setdiff1d(indices,new_samples_var,assume_unique=True)
        y_greedy = y_greedy[np.array([np.where(indice==indices)[0][0] for indice in indices_greedy])]
        new_samples_greedy = indices_greedy[torch.argsort(y_greedy.cpu())][:n_greedy]
        new_samples = np.concatenate([new_samples_var, new_samples_greedy])
        return new_samples

    def _make_model(self):
        if self.representation_model_config['model_class'] == "ActiveGNN":
            with open(self.graph_file, "rb") as f:
                data = pickle.load(f)
                self.index_map = {}  # map linear-index back to double-index
                self.index_map_rev = {}  # map double-index  back to linear-inndex
                if isinstance(data,list):
                    self.hiv_indices = data[1]

                    # The follwoing two loops is dirty solution to make sure it runs, to be removed 
                    # once the indices inconsistency is resolved 
                    # if len(self.hiv_indices)<356:
                    #     self.hiv_indices.extend(range(max(self.hiv_indices)+1,max(self.hiv_indices)+357-len(self.hiv_indices)))
                    data = data[0]
                    for idx,i_tuple in enumerate(combinations(range(len(self.hiv_indices)),2)):
                        self.index_map[idx] = i_tuple
                        self.index_map_rev[i_tuple] = idx
                else:
                    for idx,i_tuple in enumerate(combinations(range(data.num_nodes),2)):
                        self.index_map[idx] = i_tuple
                        self.index_map_rev[i_tuple] = idx
                if isinstance(data, HeteroData):
                    data = data.to_homogeneous()
                data.x = torch.eye(data.num_nodes)
                data.edge_index = data.edge_index.to(torch.int64)        

            representation_params = self.representation_model_config['params']
            model = ActiveGNN(data, node_dim=representation_params["node_dim"],
                                   embedding_d=representation_params["embedding_d"],
                                   hiv_indices=self.hiv_indices, 
                                   lambda_ewc = representation_params["lambda_ewc"],
                                   update_embedding=representation_params["update_embedding"],
                                   num_epoch_init=representation_params["num_epoch_init"],
                                   hidden_dim=representation_params["hidden_dim"], 
                                   num_conv_layers=representation_params["num_conv_layers"],
                                   reg_class = self.regression_model_config["model_class"],
                                   device = self.device
                             )
        elif self.representation_model_config['model_class'] == "ActiveFE":
            representation_params = self.representation_model_config['params']
            self.index_map = {}  # map linear-index back to double-index
            self.index_map_rev = {}  # map double-index  back to linear-inndex
            for idx,i_tuple in enumerate(combinations(range(representation_params["num_nodes"]),2)):
                self.index_map[idx] = i_tuple
                self.index_map_rev[i_tuple] = idx
            model = ActiveFE(num_nodes=representation_params["num_nodes"],
                             embedding_d=representation_params["embedding_d"],
                             update_embedding=representation_params["update_embedding"],
                              reg_class = self.regression_model_config["model_class"],
                              device = self.device
                             )
        else:
            raise NotImplementedError()        
        return model