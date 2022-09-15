import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
recommend_dir = os.path.realpath(os.path.dirname(app_path))
recommendation_dir = os.path.dirname(recommend_dir)

sys.path.insert(0, recommendation_dir)

from flask import Flask, request, jsonify
from sklearn.externals import joblib
from sklearn.utils import shuffle
import traceback
import pandas as pd
import numpy as np
from content_based import content_ae
from collaborative_filtering import collaborative_ae
import article_loader
from table_models.Artc import Articles, Categories, article_category, db
from table_models.Intr import Interactions
from keras.models import load_model
import tensorflow as tf
import logging

from datetime import date, datetime, timedelta

yesterday = date.today() - timedelta(days=2)
yesterday.strftime('%Y-%m-%d')

yesterday_time = datetime.strptime('01:01:01', '%H:%M:%S').time()
yes_datetime = datetime.combine(yesterday, yesterday_time)

months = date.today() - timedelta(days=120)
months.strftime('%Y-%m-%d')

months_time = datetime.strptime('01:01:01', '%H:%M:%S').time()
months_ago = datetime.combine(months, months_time)

ArticleSet = Articles.query.filter(Articles.status_id == 5,
                                   Articles.updated_at > months).all()

CategorySet = Categories.query.all()  # query categories

InteractionSet = Interactions.query.all()  # query interactions 

cate = db.session.query(article_category).join(Articles).all()

# store all articles in array
article_attr, Interactions_df, cate_df = article_loader. \
    article_loader(ArticleSet,
                   InteractionSet, cate)

contents_df = pd.DataFrame(article_attr,
                           columns=['id', 'title',
                                    'content', 'updated_at'])

print("number of articles", len(contents_df))

indices = pd.Series(contents_df['id'], index=contents_df['title'])

app = Flask(__name__)

model = load_model(os.path.join(recommendation_dir,
                                "collaborative_filtering/cf_model.h5"))

graph1 = tf.get_default_graph()

encoded_data = joblib.load(os.path.join(recommendation_dir,
                                        "content_based/encoded_data.pkl"))

encoded_cosine_sim = joblib.load(os.path.join(recommendation_dir,
                                              "content_based/encoded_cosine_sim.pkl"))


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        recommended_articles = []
        new_query = request.args.get("user_id")

        # Collaborative Filtering Rating Results (CF only)
        datan = Interactions_df[['user_id', 'article_id',
                                 'interaction_type_id', 'updated_at']]

        data1, X_test_array, iids_to_pred = collaborative_ae. \
            collaborative_ae. \
            user_article_recommendation(datan, new_query)

        with graph1.as_default():
            predictions = model.predict(X_test_array)
            print(predictions)

        i_sort = np.argsort(-predictions, axis=0)

        iidn = iids_to_pred[i_sort]
        iidn = pd.DataFrame(iidn, columns=['iids_to_pred'])

        print('Top items for the user has iid {0} with predicted rating {1}'
              .format(iidn, predictions))

        # Recommend similar articles to rated articles to n user (CB)

        new_query_i = datan.loc[datan['user_id'] == new_query, :]

        new_query_isort = new_query_i.sort_values(by=['updated_at'],
                                                  ascending=False).head(3)

        new_query_isorted = shuffle(new_query_isort).head(1)

        new_query_id = new_query_isorted['article_id'].values[0]

        article_cate = cate_df.loc[cate_df['article_id'] ==
                                   new_query_id, 'category_id'].values[0]

        new_query_article_index = contents_df.loc[contents_df['id'] ==
                                                  new_query_id].index.values[0]

        article_indexes_sim = content_ae. \
            content_ae.article_recommendations(new_query_article_index,
                                    encoded_data, encoded_cosine_sim, 20)

        recommended_index, recommended_titles, \
        recommended_contents, category, \
        recommended_content_date, recomm_table_sim = content_ae. \
            content_ae.similar_articles(article_indexes_sim, contents_df, cate_df)

        recomm_table_sorted = recomm_table_sim[(recomm_table_sim['updated_at'] > yes_datetime)]

        recomm_table_sorted_sim = recomm_table_sorted.loc[recomm_table_sorted['category'] == article_cate]

        if len(recomm_table_sorted_sim) < 5:
            recomm_table_sorted_sim = recomm_table_sorted.head(10)

        recommended_articles_sim = [{"Article_id": t, "Title": s, "Category": n}
                                    for t, s, n in zip(recomm_table_sorted_sim['id'],
                                                       recomm_table_sorted_sim['title'],
                                                       recomm_table_sorted_sim['category'])]

        # Recommend the unrated articles (different) to n user (CF-CB)

        new_query_id = data1.loc[data1['article'].isin(iidn['iids_to_pred'])].all(1).index.values
        article_number = data1.iloc[new_query_id]['article_id'].values
        article_number = np.unique(article_number).astype(int)

        recomm_table_diff = collaborative_ae. \
            collaborative_ae. \
            similar_user_article(contents_df,
                                 cate_df,
                                 article_number)

        recomm_table_sorted_diff = recomm_table_diff[(recomm_table_diff['updated_at'] > yes_datetime)].head(10)

        recommended_articles_diff = [{"Article_id": t, "Title": s, "Category": n}
                                     for t, s, n in zip(recomm_table_sorted_diff['id'],
                                                        recomm_table_sorted_diff['title'],
                                                        recomm_table_sorted_diff['category'])]

        recommended_articles = recommended_articles_sim + recommended_articles_diff

        return jsonify({"status": "success",
                        "message": "Successfully recommended",
                        "data": recommended_articles})

    except Exception as error:
        logging.info(error)
        return jsonify({"code": 400,
                        "status": "error",
                        "message": "invalid user-article id in interactions table",
                        "data": []})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
