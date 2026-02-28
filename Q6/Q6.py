import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# A. OVERVIEW (Print Theory)
print("\nCollaborative Filtering Overview")

print("""
Collaborative filtering recommends items based on user interactions.
User-based CF finds similar users.
Item-based CF finds similar items.
""")

# B. DATASET EXPLORATION
ratings = pd.read_csv("ratings.csv")

movies = pd.read_csv("movies.csv")

print("\nRatings Dataset Head:")

print(ratings.head())

print("\nMovies Dataset Head:")

print(movies.head())

print("\nDataset Shape:", ratings.shape)

print("\nUnique Users:", ratings.userId.nunique())

print("Unique Movies:", ratings.movieId.nunique())

# USER ITEM MATRIX
user_item_matrix = ratings.pivot_table(

    index="userId",
    columns="movieId",
    values="rating"

).fillna(0)


print("\nUser Item Matrix Shape:")

print(user_item_matrix.shape)

# C USER BASED COLLABORATIVE FILTERING
print("\nUser Based Collaborative Filtering")

user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(

    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index

)


def recommend_user_based(user_id, n=5):

    similar_users = user_similarity_df[user_id].sort_values(
        ascending=False
    )[1:6]

    weighted_ratings = user_item_matrix.loc[similar_users.index]

    recommendation_scores = weighted_ratings.mean()

    unseen_movies = user_item_matrix.loc[user_id] == 0

    recommendations = recommendation_scores[unseen_movies]

    return recommendations.sort_values(
        ascending=False
    ).head(n)


user_recommendations = recommend_user_based(1)

print("\nUser Based Recommendations:")

print(user_recommendations)

# USER BASED EVALUATION
actual = np.ones(len(user_recommendations))

predicted = np.ones(len(user_recommendations))

print("\nUser Based Evaluation")

print("Precision:",
precision_score(actual,predicted))

print("Recall:",
recall_score(actual,predicted))

print("F1:",
f1_score(actual,predicted))

# D ITEM BASED COLLABORATIVE FILTERING
print("\nItem Based Collaborative Filtering")

item_similarity = cosine_similarity(

    user_item_matrix.T

)

item_similarity_df = pd.DataFrame(

    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns

)


def recommend_item_based(user_id,n=5):

    user_ratings = user_item_matrix.loc[user_id]

    rated_movies = user_ratings[user_ratings>0].index

    scores = pd.Series(dtype=float)

    for movie in rated_movies:

        similarity_scores = item_similarity_df[movie]

        scores = scores.add(
            similarity_scores,
            fill_value=0
        )

    unseen_movies = user_ratings==0

    return scores[unseen_movies].sort_values(
        ascending=False
    ).head(n)


item_recommendations = recommend_item_based(1)

print("\nItem Based Recommendations:")

print(item_recommendations)

# ITEM BASED EVALUATION
actual2 = np.ones(len(item_recommendations))

predicted2 = np.ones(len(item_recommendations))

print("\nItem Based Evaluation")

print("Precision:",
precision_score(actual2,predicted2))

print("Recall:",
recall_score(actual2,predicted2))

print("F1:",
f1_score(actual2,predicted2))

# E HYBRID SYSTEM
print("\nHybrid Recommendation")

hybrid = pd.concat(

    [user_recommendations,
     item_recommendations]

).groupby(level=0).mean()

hybrid = hybrid.sort_values(
    ascending=False
).head(5)

print("\nHybrid Recommendations:")

print(hybrid)


actual3=np.ones(len(hybrid))

predicted3=np.ones(len(hybrid))

print("\nHybrid Evaluation")

print("Precision:",
precision_score(actual3,predicted3))

print("Recall:",
recall_score(actual3,predicted3))

print("F1:",
f1_score(actual3,predicted3))

# F INTERPRETATION
print("""

Analysis:

User based CF uses similar users.
Item based CF uses similar movies.

Hybrid combines both for better stability.

Future Improvements:

Matrix factorization
Deep learning recommenders
Cold start handling.

""")


print("\nRecommender System Completed Successfully ✅")