#Dependencies
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
recommend_dir = os.path.realpath(os.path.dirname(app_path))
recommendation_dir = os.path.dirname(recommend_dir)

sys.path.insert(0, recommendation_dir)

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Reshape, Dot, Flatten 
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Concatenate, Dense, Dropout
from sklearn.preprocessing import LabelEncoder 
from collaborative_filtering import EmbeddingLayer
from keras.layers import Add, Activation, Lambda 
#from funk_svd.funk_svd import SVD
from sklearn.metrics import mean_absolute_error
import logging
 

class collaborative_ae:
    '''
    This class provides the option of using different 
    types of ML models to build a collaborative filtering 
    based recommendation system.
    It includes two main techniques which are SVD and 
    Deep Neural Net based methods (with 3 variants)  
    '''
    
    def __init__(self, data, learning_rate, regularization, n_epochs, n_factors, frac):
        self.data = data
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.frac = frac

    @staticmethod
    def funk_collaborative( data, learning_rate, regularization, n_epochs, 
                           n_factors, frac, train, val, test):
        
        '''
        funkSVD based collaborative filtering. To use this method,
        define the parameters (learning_rate, regularization, 
        n_epochs, n_factors) to initialize the model. 
        run svd.fit to train the model with the training data.
        svd.predict is for predicting the user ratings. 
        
        '''
        ######### Apply Funk SVD ##################
        svd = SVD(learning_rate=learning_rate, 
                  regularization= regularization, 
                  n_epochs=n_epochs,n_factors=n_factors)
        
        svd.fit(X=train,X_val=val, early_stopping=True,shuffle=False)
        
        pred = svd.predict(test)
        
        mae = mean_absolute_error(test['rating'], pred)
        print('Test MAE: {:.2f}'.format(mae))
        
        return svd

    @staticmethod
    def RecommenderNet1(n_users, n_articles, n_factors):
        '''
        Deep Neural Net based collaborative filtering. 
        Inputs: user and item embeddings, number of factors 
        Output: user_item relations model to predict user ratings
        
        '''
        user = Input(shape=(1,))
        
        u = Embedding(n_users, n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(user)
        u = Reshape((n_factors,))(u)
        
        article = Input(shape=(1,))
        m = Embedding(n_articles, n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(article)
        
        m = Reshape((n_factors,))(m)
        
        x = Dot(axes=1)([u, m])
        model = Model(inputs=[user, article], outputs=x)
        
        opt = Adam(lr=0.001)
        
        model.compile(loss='mean_squared_error', optimizer=opt)
        
        return model
    
    @staticmethod
    def RecommenderNet2(n_users, n_articles, n_factors, min_rating, max_rating):
        
        user = Input(shape=(1,))
        u = EmbeddingLayer.EmbeddingLayer(n_users, n_factors)(user)
        ub = EmbeddingLayer.EmbeddingLayer(n_users, 1)(user)
        
        article = Input(shape=(1,))
        
        m = EmbeddingLayer.EmbeddingLayer(n_articles, n_factors)(article)
        mb = EmbeddingLayer.EmbeddingLayer(n_articles, 1)(article)
        
        x = Dot(axes=1)([u, m])
        x = Add()([x, ub, mb])
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        
        model = Model(inputs=[user, article], outputs=x)
        
        opt = Adam(lr=0.001)
        
        model.compile(loss='mean_squared_error', optimizer=opt)
        
        return model
    
    @staticmethod
    def RecommenderDeepNet(n_users, n_articles, n_factors, min_rating, max_rating):
        user = Input(shape=(1,))
        u = EmbeddingLayer.EmbeddingLayer(n_users, n_factors)(user)
        
        article = Input(shape=(1,))
        m = EmbeddingLayer.EmbeddingLayer(n_articles, n_factors)(article)
        
        x = Concatenate()([u, m])
        x = Dropout(0.05)(x)
        
        x = Dense(20, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        x = Dense(1, kernel_initializer='he_normal')(x)
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        model = Model(inputs=[user, article], outputs=x)
        
        opt = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)
        
        return model

    @staticmethod
    def RecommenderSimplex(n_users, n_items, n_factors):
        user = Input(shape=[1], name="User-Input")
        
        u = Embedding(n_users+1, n_factors, name="User-Embedding")(user)
        
        user_vec = Flatten(name="Flatten-Users")(u)
        
        article = Input(shape=[1], name="Item-Input")
        m = Embedding(n_items+1, n_factors, name="Article-Embedding")(article)
        
        article_vec = Flatten(name="Flatten-Article")(m)
        
        prod = Dot(name="Dot-Product", axes=1)([user_vec, article_vec])
        model = Model([user, article], prod)
        model.compile('adam', 'mean_squared_error')
        
        return model
 
    @staticmethod
    def user_article_recommendation(datan, new_query):

        """
        This module is for recommending the articles that haven't been
        read by the user. It first transforms the original user id and its
        corresponding article id into a compact form using label encoding.
        Then, it searches through the indexes to find articles yet to be read
        by the user.
        The output from this serves as input to the collaborative filtering model
        e.g Deep Neural Net based method to predict the ratings.

        """
        
        user_enc = LabelEncoder() 
        user_enc_model = user_enc.fit(datan['user_id'].values)
        datan['user'] = user_enc_model.transform(datan['user_id'].values)
     
        item_enc = LabelEncoder()
        item_enc_model = item_enc.fit(datan['article_id'].values)

        datan['article'] = item_enc_model.transform(datan['article_id'].values)
        datan['interaction_type_id'] = datan['interaction_type_id'].values.astype(np.float32)
        iids = datan['article'].unique()
        
        new_query_iid = datan.loc[datan['user_id'] == str(new_query), 'article']
        new_query_uid = datan.loc[datan['user_id'] == str(new_query)].index.values
        
        iids_to_pred = np.setdiff1d(iids, new_query_iid)
        new_query_id = datan['user'][new_query_uid].values
        
        test_set = [[new_query_id[0], iid, 2] for iid in iids_to_pred]
        test_set_df = pd.DataFrame(test_set, columns=['user', 'article', 'interaction_type_id'])
        test_set_df['interaction_type_id'] = test_set_df['interaction_type_id'].values.astype(np.float32)
        
        min_rating = min(test_set_df['interaction_type_id'])
        max_rating = max(test_set_df['interaction_type_id'])

        X = test_set_df[['user', 'article']].values
        X_test_array = [X[:, 0], X[:, 1]]
        
        return datan, X_test_array, iids_to_pred

    @staticmethod
    def similar_user_article(contents_df, cate_df, article_number):
                
        recommended_titles = []       
        recommended_contents = []         
        recommended_index = []       
        recommended_content_date = []
        category = []
        
        for i in article_number:  
            try:
                
                indx = contents_df[contents_df['id'] == i].index.values[0]
                recommended_titles.append(contents_df['title'][indx])      
                recommended_contents.append(contents_df['content'][indx])            
                recommended_index.append(int(contents_df['id'][indx]))
                recommended_content_date.append(contents_df['updated_at'][indx])             
                category.append(cate_df['category_id'][indx])
            except Exception as error: 
                logging.info(error)  
                continue
        
        recomm_table = pd.DataFrame({'id': recommended_index, 'title': recommended_titles, 
                                     'content': recommended_contents, 'category': category, 
                                     'updated_at':recommended_content_date})
    
        return recomm_table
