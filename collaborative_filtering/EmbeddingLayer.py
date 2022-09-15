from keras.layers import Reshape
from keras.layers.embeddings import Embedding 
from keras.regularizers import l2 
 

class EmbeddingLayer:
    '''
    To construct user or article embedding with x number of factors
    '''

    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x
