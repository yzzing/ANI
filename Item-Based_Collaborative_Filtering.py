# https://www.kaggle.com/code/varian97/item-based-collaborative-filtering

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# Read the anime.csv
anime = pd.read_csv("anime.csv")
# print(anime.head())

# Select TV, Movie, OVA, ONA
anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie') | (anime['type'] == 'OVA') | (anime['type'] == 'ONA')]

# Select only famous anime, 75% percentile
m = anime['members'].quantile(0.75)
anime = anime[(anime['members'] >= m)]

# Read the anime.csv
rating = pd.read_csv("rating.csv")
# print(rating.head())


# Replace missing rating with NaN
rating.loc[rating.rating == -1, 'rating'] = np.NaN
# print(rating.head())


# Index for anime name
anime_index = pd.Series(anime.index, index=anime.name)
# print(anime_index.head())


# Join the data
joined = anime.merge(rating, how='inner', on='anime_id')
# print(joined.head())


# Create a pivot table
joined = joined[['user_id', 'name', 'rating_y']]

pivot = pd.pivot_table(joined, index='name', columns='user_id', values='rating_y')
# print(pivot.head())


# Drop all users that never rate an anime
pivot.dropna(axis=1, how='all', inplace=True)
# print(pivot.head())


# Center the mean around 0 (centered cosine / pearson)
pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
# print(pivot_norm.head())


# Item Based Collaborative Filtering
# fill NaN with 0
pivot_norm.fillna(0, inplace=True)
# print(pivot_norm.head())


# Calculate Similar Items
# convert into dataframe to make it easier
item_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index, columns=pivot_norm.index)
# print(item_sim_df.head())

def get_similar_anime(anime_name):
    if anime_name not in pivot_norm.index:
        return None, None
    else:
        sim_animes = item_sim_df.sort_values(by=anime_name, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=anime_name, ascending=False).loc[:, anime_name].tolist()[1:]
        return sim_animes, sim_score


# Helper Function
# predict the rating of anime x by user y
def predict_rating(user_id, anime_name, max_neighbor=10):
    animes, scores = get_similar_anime(anime_name)
    anime_arr = np.array([x for x in animes])
    sim_arr = np.array([x for x in scores])
    
    # select only the anime that has already rated by user x
    filtering = pivot_norm[user_id].loc[anime_arr] != 0
    
    # calculate the predicted score
    s = np.dot(sim_arr[filtering][:max_neighbor], pivot[user_id].loc[anime_arr[filtering][:max_neighbor]]) \
            / np.sum(sim_arr[filtering][:max_neighbor])
    
    return s


# Get Recommendation
# recommend top n_anime for user x based on item collaborative filtering algorithm
def get_recommendation(user_id, n_anime=10):
    print("For user", user_id, ",")
    predicted_rating = np.array([])
    
    for _anime in pivot_norm.index:
        predicted_rating = np.append(predicted_rating, predict_rating(user_id, _anime))
    
    # don't recommend something that user has already rated
    temp = pd.DataFrame({'predicted':predicted_rating, 'name':pivot_norm.index})
    filtering = (pivot_norm[user_id] == 0.0)
    temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

    # recommend n_anime anime
    return anime.loc[anime_index.loc[temp.name[:n_anime]]]

print(get_recommendation(3))
