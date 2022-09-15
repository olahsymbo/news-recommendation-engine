# Dependencies#
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
recommend_dir = os.path.realpath(os.path.dirname(app_path))
recommendation_dir = os.path.dirname(recommend_dir)

sys.path.insert(0, recommendation_dir)
import numpy as np
import pandas as pd
import pickle
from content_based import content_ae
from collaborative_filtering import collaborative_ae
from recommender_api import article_loader
from table_models.Artc import Articles, Categories, article_category, db
from table_models.Intr import Interactions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datetime import date, datetime, timedelta

yesterday = date.today() - timedelta(days=2)
yesterday.strftime('%Y-%m-%d')

yesterday_time = datetime.strptime('01:01:01', '%H:%M:%S').time()
yes_datetime = datetime.combine(yesterday, yesterday_time)

months = date.today() - timedelta(days=120)
months.strftime('%Y-%m-%d')

months_time = datetime.strptime('01:01:01', '%H:%M:%S').time()
months_ago = datetime.combine(months, months_time)

# ArticleSet = Articles.query.filter_by(status_id=7).all() # query all articles

ArticleSet = Articles.query.filter(Articles.status_id == 5,
                                   Articles.updated_at > months).all()

CategorySet = Categories.query.all()  # query categories

InteractionSet = Interactions.query.all()  # query interactions 

cate = db.session.query(article_category).join(Articles).all()

# store all articles in array
article_attr, Interactions_df, cate_df = article_loader. \
    article_loader(ArticleSet,
                   InteractionSet, cate)

# Deep Learning for Content Based

# store all articles contents in array
contents_df = pd.DataFrame(article_attr, columns=['id', 'title', 'content', 'updated_at'])
print("Total number of articles in db", len(contents_df))

indices = pd.Series(contents_df['id'], index=contents_df['title'])

# prepare for content based learning
encoding_dim = 200

encoded_data, encoded_cosine_sim = content_ae. \
    content_ae. \
    content_mf(contents_df,
               encoding_dim,
               compression_factor=None)

pickle.dump(encoded_data, open(os.path.join(recommendation_dir, "content_based/encoded_data.pkl"), "wb"))

pickle.dump(encoded_cosine_sim, open(os.path.join(recommendation_dir, "content_based/encoded_cosine_sim.pkl"), "wb"))

# prepare for collaborative based learning
article_data = article_attr

data = Interactions_df[['user_id', 'article_id', 'interaction_type_id', 'updated_at']]

# Deep Learning for Collaborative
user_enc = LabelEncoder()
user_enc_model = user_enc.fit(data['user_id'].values)
pickle.dump(user_enc_model, open(os.path.join(recommendation_dir, "collaborative_filtering/user_enc_model.pkl"), "wb"))

data['user'] = user_enc_model.transform(data['user_id'].values)
n_users = data['user'].nunique()

item_enc = LabelEncoder()
item_enc_model = item_enc.fit(data['article_id'].values)
pickle.dump(item_enc_model, open(os.path.join(recommendation_dir,
                                              "collaborative_filtering/item_enc_model.pkl"), "wb"))

data['article'] = item_enc_model.transform(data['article_id'].values)
n_article = data['article'].nunique()

data['interaction_type_id'] = data['interaction_type_id'].values.astype(np.float32)
min_rating = min(data['interaction_type_id'])
max_rating = max(data['interaction_type_id'])
n_users, n_article, min_rating, max_rating

n_users = data['user'].nunique()
n_article = data['article'].nunique()
min_rating = min(data['interaction_type_id'])
max_rating = max(data['interaction_type_id'])
n_factors = 50

X = data[['user', 'article']].values
y = data['interaction_type_id'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

model = collaborative_ae.collaborative_ae.RecommenderNet2(n_users, n_article, n_factors, min_rating, max_rating)
model.summary()

history = model.fit(x=X_train_array, y=y_train, batch_size=20, epochs=20,
                    verbose=1, validation_data=(X_test_array, y_test))

predictions = model.predict(X_test_array)
data1 = user_enc.fit_transform(predictions)
model.save(os.path.join(recommendation_dir, 'collaborative_filtering/cf_model.h5'))
