from abc import abstractmethod, ABC
import numpy as np
import pandas as pd

# TODO: 
#   finish defining this abstract class
class ActiveRepresentationModel(ABC):
    """
    Class to represent models that support active representation learning
    All subclasses must implement the following:
        1. A model that contains encoder and decoder 
        2. regression network 
        3. 'train' method, which initialize the model by training the encoder-decoder network 
    """
    def __init__(self, model_config=None, name="ActiveRepresentationLearning",**kwargs):
        self.name = name
        super().__init__(**kwargs)

    @abstractmethod
    def train(self):
        """
        Function to train the encoder-decoder network 
        Parameters:
    
        """
        pass

    @abstractmethod
    def train_joint(self):
        """
        Function to train jointly the encoder-decoder network and regression network
        Parameters:

        """
        pass