import pandas as pd


def article_loader(ArticleSet, InteractionSet, cate):
    article_attr = []

    for Article in ArticleSet:
        cont = [Article.id, Article.title, Article.content, Article.updated_at]
        article_attr.append(cont)

    # store all interactions in array
    Interaction_attr = []

    for Interaction in InteractionSet:
        Intr_cont = [Interaction.id, Interaction.user_id,
                     Interaction.article_id, Interaction.interaction_type_id,
                     Interaction.created_at, Interaction.updated_at]
        Interaction_attr.append(Intr_cont)

    Interactions_df = pd.DataFrame(Interaction_attr,
                                   columns=['id', 'user_id',
                                            'article_id', 'interaction_type_id',
                                            'created_at', 'updated_at'])

    cate_arte = []

    for row in cate:
        cateart = [row[0], row[1]]
        cate_arte.append(cateart)

    cate_df = pd.DataFrame(cate_arte, columns=['article_id', 'category_id'])

    return article_attr, Interactions_df, cate_df
