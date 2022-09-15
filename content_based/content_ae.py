# Dependencies
import inspect
import os

app_path = inspect.getfile(inspect.currentframe())
module_dir = os.path.realpath(os.path.dirname(app_path))
import sys

sys.path.insert(0, module_dir)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import pandas as pd


# import read_data
class content_ae:

    def __init__(self, content_df, encoding_dim, compression_factor=None):
        self.contents_df = content_df
        self.encoding_dim = encoding_dim
        self.compression_factor = compression_factor

    @staticmethod
    def content_mf(contents_df, encoding_dim, compression_factor):
        '''
        This module is for training the content based recommendation part of the engine.
            
        '''

        max_features = 3000
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                             min_df=0, stop_words='english', max_features=max_features)
        tfidf_matrix = tf.fit_transform(contents_df['content'])
        tfidf_matrix.shape

        input_dim = tfidf_matrix.shape[1]
        encoding_dim = encoding_dim
        compression_factor = float(input_dim / encoding_dim)

        autoencoder = Sequential()
        autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
        autoencoder.add(Dense(input_dim, activation='sigmoid'))

        input_img = Input(shape=(input_dim,))
        encoder_layer = autoencoder.layers[0]
        encoder = Model(input_img, encoder_layer(input_img))

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(tfidf_matrix, tfidf_matrix, epochs=20, batch_size=100, shuffle=True)

        encoded_data = encoder.predict(tfidf_matrix)
        decoded_data = autoencoder.predict(tfidf_matrix)
        encoded_cosine_sim = cosine_similarity(encoded_data, encoded_data)

        return encoded_data, encoded_cosine_sim

    # Recommend for each article
    @staticmethod
    def article_recommendations(idx, encoded_data, encoded_cosine_sim, top_articles):
        sim_scores = list(enumerate(encoded_cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_articles]

        article_indices = [i[0] for i in sim_scores]

        return article_indices

    @staticmethod
    def similar_articles(article_indexes_sim, contents_df, cate_df):
        recommended_titles = []
        recommended_contents = []
        recommended_index = []
        recommended_content_date = []
        category = []

        for i in article_indexes_sim:
            recommended_titles.append(contents_df['title'][i])
            recommended_contents.append(contents_df['content'][i])
            recommended_index.append(int(contents_df['id'][i]))
            recommended_content_date.append(contents_df['updated_at'][i])
            indx = int(contents_df['id'][i])
            category.append(int(cate_df.loc[cate_df['article_id'] == indx, 'category_id'].values[0]))

        recomm_table = pd.DataFrame({'id': recommended_index, 'title': recommended_titles,
                                     'content': recommended_contents, 'category': category,
                                     'updated_at': recommended_content_date})

        return recommended_index, recommended_titles, recommended_contents, \
               category, recommended_content_date, recomm_table
