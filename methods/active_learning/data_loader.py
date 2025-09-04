
import numpy as np 
from numpy.random import default_rng

import pandas as pd 
import matplotlib.pyplot as plt

class DataLoader():
    """
    Loads gene-gene interaction matrix
    """
    def __init__(self, synthetic=False, **kwargs):
        if synthetic:
            self.load(synthetic=True, **kwargs)

    def load(self, synthetic=False, data_path=None, **kwargs):
                          
        if synthetic:
            # update this model to reflect the BMF with side information
            self.random_low_rank_matrix = np.random.normal(size=[kwargs['rank'], kwargs['dim']]) 
            self.interaction_matrix = pd.DataFrame(np.dot(self.random_low_rank_matrix.T,  self.random_low_rank_matrix))  + np.random.normal(0, .1, size=[kwargs['dim'], kwargs['dim']]) # * (1-np.eye(data_dim)))
        else:
       
            df_orig = pd.read_csv(data_path, sep=kwargs['sep'], 
                                                  index_col=kwargs['index_col'], 
                                                  header=kwargs['header'])
            self.raw_data = pd.DataFrame(df_orig)

            if kwargs['log_transform']:
                minvalue = df_orig.min().min()
                if minvalue>0:
                    self.interaction_matrix = pd.DataFrame(np.log(df_orig))
                else:
                    self.interaction_matrix = pd.DataFrame(np.log(df_orig - minvalue + 1e-10))

            else:
                self.interaction_matrix = self.raw_data


    def load_random_subset_data(self, dim):
        # randomly sample rows/cols using data_dim        
        if dim < self.raw_data.shape[0]:
            subset = np.sort(np.random.choice(self.interaction_matrix.shape[0], replace=False, size=dim))
            self.load_subset_data(subset)
            self.subset = subset

    def load_subset_data(self, subset):
        self.interaction_matrix = self.interaction_matrix.iloc[subset,subset]

    def get_data(self):
        return self.interaction_matrix

    def get_noisy_data(self, noise=1):
        return self.interaction_matrix.copy() + np.random.normal(0, noise, size=self.interaction_matrix.shape)

    def plot_data(self):
        plt.matshow(self.interaction_matrix)
        plt.colorbar()
        plt.show()
        
        
class GordonDataLoader(DataLoader):
    """
    Loads gene-gene interaction matrix with multiple experiments
    Due to the presence of multiple experiments, this class operates primarily in "index format", where a dataframe
    contains gene 1, gene2, and value columns. As opposed to a matrix format, where gene 1 and gene2 represent names
    of rows and columns respectively.
    """
            
    def load(self, data_path='/usr/workspace/PROTECT/GordonHIVData.csv', **kwargs):
    
        self.raw_data = pd.read_csv(data_path)

        # Sort gene pairs alphabetically to create a dataframe corresponding to upper triangular matrix (not a metrix itself)
        data_df_triu = self.raw_data.copy()
        mask = data_df_triu['Name_Gene1'] > data_df_triu['Name_Gene2']
        data_df_triu.loc[mask, ['Name_Gene1', 'Name_Gene2']] = data_df_triu.loc[mask, ['Name_Gene2', 'Name_Gene1']].values
        data_df_triu.loc[mask, ['Entrez_Gene1', 'Entrez_Gene2']] = data_df_triu.loc[mask, ['Entrez_Gene2', 'Entrez_Gene1']].values

        # remove invalid entries, where viral load is below zero
        data_df_triu_clean = data_df_triu[data_df_triu['ViralIntens']>0]

        if kwargs['log_transform']:

            minvalue = data_df_triu_clean['ViralIntens'].min()
            if minvalue>0:
                data_df_triu_clean.loc[:, ['LogViralIntens']] = np.log(data_df_triu_clean['ViralIntens'])
            else:
                data_df_triu_clean.loc[:, ['LogViralIntens']] = np.log(data_df_triu_clean['ViralIntens'] - minvalue + 1e-10)
            if kwargs['center_data']:
                # center data after log
                data_df_triu_clean.loc[:, ['LogViralIntens']] = data_df_triu_clean['LogViralIntens'] - data_df_triu_clean['LogViralIntens'].mean()

            self.val_column = 'LogViralIntens'

        else:
            self.val_column = 'ViralIntens'

        # Assign numeric IDs to genes
        gene_names = np.unique(np.append(data_df_triu_clean['Name_Gene1'],data_df_triu_clean['Name_Gene2']))        
        self.n_genes = len(gene_names)
        #gene_names = np.sort(data_df_triu_clean['Name_Gene1'].unique())
        numeric_ids = np.arange(len(gene_names))
        self.name_to_numeric = dict(zip(gene_names, numeric_ids))
        self.numeric_to_name = dict(zip(numeric_ids, gene_names))
        data_df_triu_clean.loc[:,['gene_id1']] = data_df_triu_clean['Name_Gene1'].map(self.name_to_numeric)
        data_df_triu_clean.loc[:,['gene_id2']] = data_df_triu_clean['Name_Gene2'].map(self.name_to_numeric)            
            
        self.observations = data_df_triu_clean.copy()
        # Following matrices are produced by mean aggregation
        self.interaction_triu_matrix = data_df_triu_clean.pivot_table(index='Name_Gene1', columns='Name_Gene2', values=self.val_column, aggfunc='mean')
        self.interaction_matrix = self.interaction_triu_matrix.fillna(0) + self.interaction_triu_matrix.fillna(0).T - np.diag(np.diag(self.interaction_triu_matrix))
        
            
    def load_random_subset_data(self, num_samples = None, percent_missing = None):
        if num_samples is None and percent_missing is not None:
            # Select random samples for training. This is done instead of masking.
            num_samples = int(len(self.observations)*(1.0-percent_missing))
        elif num_samples is None:
            raise("You must specify either number of samples or percentage of data missing from the training set.")
        
        print("{}/{} samples selected for training".format(num_samples, len(self.observations)))
        
        train_ids = np.random.choice(len(self.observations),size=num_samples,replace=False)
        test_ids = np.setdiff1d(np.arange(len(self.observations)), train_ids)

        self.trainset = self.observations.iloc[train_ids,:]
        self.testset = self.observations.iloc[test_ids,:]
                                
        self.trainX = self.trainset[['gene_id1','gene_id2']]
        self.trainY = self.trainset[self.val_column]
                                
        self.testX = self.testset[['gene_id1','gene_id2']]
        self.testY = self.testset[self.val_column]
        
    def bootstrap_sample(self, X, size=2, if_mean=True):
        '''
        Bottstrap observations from original data
        '''
        
        def random_element(values):
            # return np.mean(np.random.choice(values,size=size,replace=False))
            return np.mean(np.random.choice(values,size=size,replace=False)) if if_mean else np.random.choice(values,size=2,replace=False)
        
        # order gene_ids in each row
        _X = np.sort(X, axis=1)
        return np.array([self.observations.loc[(self.observations['gene_id1'] == row[0]) & (self.observations['gene_id2'] == row[1]), self.val_column].agg(random_element) for row in _X])
    
    def bootstrap_sample_seeded(self, X, if_mean=True):
        '''
        Bottstrap observations from original data
        '''
        def random_element(values, if_mean=True):
            seed = 0 
            rng =  default_rng(seed)
            return np.mean(rng.choice(values,size=2,replace=False)) if if_mean else rng.choice(values,size=2,replace=False)
        
        # order gene_ids in each row
        _X = np.sort(X, axis=1)
        return np.array([self.observations.loc[(self.observations['gene_id1'] == row[0]) & (self.observations['gene_id2'] == row[1]), self.val_column].agg(random_element) for row in _X], if_mean=if_mean)

    
    #[self.observations.loc[(self.observations['gene_id1'] == row[0]) & (self.observations['gene_id2'] == row[1]), self.val_column]
    # Out DF:
    # gene_id1 gene_id2 LogViralLoad
    # A.       B.        0.1
    # A.       B.        -5

if __name__ == '__main__':
    n_genes=356 

    dl = GordonDataLoader()
    dl.load(log_transform=True, center_data=False)
    allX = np.array(np.triu_indices(n_genes,1)).T
    interaction_vector_replicates = dl.bootstrap_sample(allX, size=4, if_mean=False)
    print(interaction_vector_replicates.shape)
    interaction_vector = np.mean(interaction_vector_replicates, axis=1)
    print(interaction_vector.shape)