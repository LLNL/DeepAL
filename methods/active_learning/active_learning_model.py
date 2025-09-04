from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from torch.nn import Module

class ActiveLearningModel(Module, ABC):
    """
    Class to represent models that support active learning
    All subclasses must implement a 'fit' method, which fits a model on
    collected / observed data
    """
    def __init__(self, model_config=None, name="ActiveLearning", **kwargs):
        self.name = name
        super().__init__(**kwargs)

    @abstractmethod
    def fit(self, all_targets, seen_mask):
        """
        Function to fit a probabilistic model given previously observed data
        Parameters:
            - all_targets: vector containing the universe of targets to sample
            - seen_mask: a boolean mask of size N, where N is the size of
                         the universe (i.e., all possible observations), and
                         the i-th entry is one if and only if the i-th point
                         has already been collected
        """
        pass

    # TODOs:
    #      1. cleaner solutioin to take in rounds as input
    #      2. test out different way use round information
    #         to develope adaptive schemes
    def acquisition(self, n_samples, seen_mask, n_round=0, method='thompson'):
        """
        acquisition function to choose the next set of observations to sample
        this should be called only after fitting the model
        Parameters:
            - n_samples: how many new samples to acquire
            - seen_mask: a boolean mask of size N, where N is the size of
                         the universe (i.e., all possible observations), and
                         the i-th entry is one if and only if the i-th point
                         has already been collected
            - method: choose from
                - thompson: get a sample from a posterior distribution and choose
                            the best points in this sample
                - variance: choose points with the largest model uncertainty
                - max_ent:  choose points with the largest entropy
                - optimism: chooses points that minimize E[y] - Var[y] or
                            maximize E[y] + Var[y]
                - random:   choose points uniformly at random
        """
        if method == 'thompson':
            return self.acquisition_thompson(n_samples, seen_mask)
        elif method == 'mean':
            return self.acquisition_mean(n_samples, seen_mask)
        elif method == 'variance':
            return self.acquisition_var(n_samples, seen_mask, n_round=n_round)
        elif method == 'max_ent':
            return self.acquisition_maxent(n_samples, seen_mask)
        elif method == 'optimism':
            return self.acquisition_optimism(n_samples, seen_mask)
        elif method == 'random':
            return self.acquisition_random(n_samples, seen_mask)
        elif method =="pessimism":
            return self.acquisition_pessimism(n_samples, seen_mask)
        elif method =="hybrid":
            return self.acquisition_hybrid(n_samples, seen_mask,n_round=n_round)
        elif method == "optimism-hall":
            return self.acquisition_optimism_hall(n_samples, seen_mask)
        elif method == "optimism-div":
            return self.acquisition_optimism_div(n_samples, seen_mask,n_round=n_round)
        elif method == "eps-greedy":
            return self.acquisition_eps_greedy(n_samples, seen_mask,n_round=n_round)
        elif method == "greedy-div":
            return self.acquisition_greedy_div(n_samples, seen_mask,n_round=n_round)
        elif method =="greedy-elim-div":
            return self.acquisition_greedy_elim_div(n_samples, seen_mask,n_round=n_round)
        elif method =='greedy-explore':
            return self.acquisition_greedy_explore(n_samples, seen_mask, n_round=n_round)
        elif method =='bait':
            return self.acquisition_bait(n_samples, seen_mask, n_round=n_round)
        elif method =='variance_mixed':
            return self.acquisition_variance_mixed(n_samples, seen_mask, n_round=n_round)
        elif method =='diag': # placeholder for the diag baseline method
            return 
        elif method =="annealing":
            return self.acquisition_annealing(n_samples, seen_mask, n_round=n_round)
        elif method == "diversification":
            return self.acquisition_diversification(n_samples, seen_mask, n_round=n_round)
        elif method == "max-optimism":
            return self.acquisition_max_optimism(n_samples, seen_mask, n_round=n_round)
        elif ".csv" in method:
            return self.acquisition_CSV(n_samples, seen_mask, method)
        elif method == 'badge':
            return self.acquisition_badge(n_samples, seen_mask, n_round=n_round)
        else:
            raise ValueError("Unrecognized method %s" % method)
        
    def acquisition_max_optimism(n_samples, seen_mask, n_round):
        raise NotImplementedError()

    def acquisition_badge(n_samples, seen_mask, n_round):
        raise NotImplementedError()

    def acquisition_thompson(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acqusition_mean(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_var(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_maxent(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_optimism(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_pessimism(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_hybrid(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_optimism_hall(self, n_samples, seen_mask):
        raise NotImplementedError()

    def acquisition_optimism_div(self, n_samples, seen_mask):
        raise NotImplementedError()
    
    def acquisition_acquisition_eps_greedy(self, n_samples, seen_mask, n_round):
        raise NotImplementedError()
    
    def acquisition_acquisition_greedy_div(self, n_samples, seen_mask, n_round):
        raise NotImplementedError()
    
    def acquisition_greedy_elim_div(self, n_samples, seen_mask, n_round):
        raise NotImplementedError()
    
    def acquisition_greedy_explore(self, n_samples, seen_mask, n_round):
        raise NotImplementedError()
    
    def acquisition_bait(self, n_samples, seen_mask, n_round):
        raise NotImplementedError()
    
    def acquisition_variance_mixed(n_samples, seen_mask, n_round):
        raise NotImplementedError()
    
    def acquisition_annealing(self, n_samples, seen_mask, n_round):
        raise NotImplementedError()

    def acquisition_random(self, n_samples, seen_mask):
        """
        Choose n_samples uniformly at random from the points that haven't been seen yet
        (i.e., seen_mask == False)
        """
        indices = np.arange(len(seen_mask))
        return np.random.choice(indices[~seen_mask], size=n_samples, replace=False)

    def acquisition_CSV(self, n_samples, seen_mask, method):
        df = pd.read_csv(method, header = None, delimiter =',')
        df = df.iloc[: , 1:]

        values = df.values 
        print(len(values))
        return np.array(values)

    def is_updatable(self):
        method = getattr(self, "update", None)
        return callable(method)