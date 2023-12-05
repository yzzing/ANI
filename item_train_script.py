import numpy as np
import pandas as pd
import warnings
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    input_data_path = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
    # Read the anime.csv
    anime = pd.read_csv(os.path.join(input_data_path, 'anime.csv'))
    # print(anime.head())


    # Preprocessing
    # Select TV, Movie, OVA, ONA
    anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]

    # Select only famous anime, 75% percentile
    f = anime['members'].quantile(0.75)
    anime = anime[(anime['members'] >= f)]

    # Read the anime.csv
    rating = pd.read_csv(os.path.join(input_data_path, "rating.csv"))
    # print(rating.head())


    # Replace missing rating with NaN
    rating.loc[rating.rating == -1, 'rating'] = np.NaN
    # print(rating.head())

    # import matplotlib.pyplot as plt

    # Index for anime name
    anime_ind = pd.Series(anime.index, index=anime.name)

    # Join the data
    join = anime.merge(rating, how='inner', on='anime_id')

    # Create a pivot table
    join = join[['user_id', 'name', 'rating_y']]
    pivot = pd.pivot_table(join, index='name', columns='user_id', values='rating_y')

    # Drop all users who has no rating
    pivot.dropna(axis=1, how='all', inplace=True)

    # Center the mean around 0 (centered cosine / pearson)
    pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)

    # Drop all users who has no rating
    pivot.dropna(axis=1, how='all', inplace=True)
    # print(pivot.head())


    # Center the mean around 0 (centered cosine / pearson)
    pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
    # print(pivot_norm.head())


    # Item Based Collaborative Filtering
    # Fill NaN with 0
    pivot_norm.fillna(0, inplace=True)
    # print(pivot_norm.head())


    # Calculate Similar Animes
    # Convert into dataframe
    item_similar_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index, columns=pivot_norm.index)
    # print(item_sim_df.head())

    def get_similar_anime(anime_name):
        if anime_name not in pivot_norm.index:
            return None, None
        else:
            similar_animes = item_similar_df.sort_values(by=anime_name, ascending=False).index[1:]
            similar_score = item_similar_df.sort_values(by=anime_name, ascending=False).loc[:, anime_name].tolist()[1:]
            return similar_animes, similar_score


    # Predict the rating of anime x by user y
    def predict_rating(user_id, anime_name, max_neighbor=10):
        animes, scores = get_similar_anime(anime_name)
        anime_arr = np.array([x for x in animes])
        sim_arr = np.array([x for x in scores])
        
        # Select the anime that has already rated by user x
        filtering = pivot_norm[user_id].loc[anime_arr] != 0
        
        # Calculate the predicted score
        predicted_score = np.dot(sim_arr[filtering][:max_neighbor], pivot[user_id].loc[anime_arr[filtering][:max_neighbor]]) \
                / np.sum(sim_arr[filtering][:max_neighbor])
        
        return predicted_score


    # Recommendation
    # Recommend the top n number of anime for user by item-based collaborative filtering algorithm
    def get_recommendation(user_id, n=10):
        predicted_rating = np.array([])
        
        for _anime in pivot_norm.index:
            predicted_rating = np.append(predicted_rating, predict_rating(user_id, _anime))
        
        # Exclude the anime that already rated by user
        temp = pd.DataFrame({'predicted':predicted_rating, 'name':pivot_norm.index})
        filtering = (pivot_norm[user_id] == 0.0)
        temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

        # Recommend n number of anime
        return anime.loc[anime_ind.loc[temp.name[:n]]]
    
    result = get_recommendation(7777)
    print(result)

    # 저장
    output_model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    recomm_list_filename = os.path.join(output_model_path, 'Recommendation_list.joblib')
    item_similar_filename = os.path.join(output_model_path, 'Item_similarity.joblib')
    
    joblib.dump(result, recomm_list_filename)
    joblib.dump(item_similar_df, item_similar_filename)



# Reference: https://www.kaggle.com/code/varian97/item-based-collaborative-filtering
