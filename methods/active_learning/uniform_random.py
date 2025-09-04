import numpy as np

from methods.active_learning.active_learning_model import ActiveLearningModel


class UniformRandom(ActiveLearningModel):
    """
        Naive baseline model that simply returns samples chosen uniformly
        at random
    """
    def __init__(self, model_config=None, name="random",**kwargs):
        super().__init__(name=name,**kwargs)
        self.filter_mask = None

    def fit(self, all_targets, seen_mask):
        pass

    def predict(self, indices):
        return np.random.permutation(len(indices))

    def acquisition(self, n_samples, seen_mask, method='thompson'):
        if method != 'random':
            raise ValueError("This model only supports 'random' acquisition")
        return self.acquisition_random(n_samples, seen_mask)
    
    def acquisition_random(self, n_samples, seen_mask):
        if self.filter_mask is None:
            indices = np.arange(len(seen_mask))[~seen_mask]
        else:
            indices = np.arange(len(seen_mask))[(~seen_mask)*self.filter_mask]
        return np.random.choice(indices, size=n_samples, replace=False)