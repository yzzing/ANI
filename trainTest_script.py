#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import numpy as np
import joblib


# In[ ]:


# SageMaker 환경에서는 학습 데이터가 이 경로에 저장됩니다.
input_data_path = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')

def precision_at_k(recommended_list, actual_list, k=5):
    if not recommended_list or not actual_list:
        return 0.0
    act_set = set(actual_list)
    rec_set = set(recommended_list[:k])
    precision = len(act_set & rec_set) / float(k)
    return precision


def load_data():
    # S3 경로에서 데이터를 로드하는 코드를 추가합니다.
    # 여기서는 로컬 파일 시스템 경로를 예시로 사용합니다.
    anime_df = pd.read_csv(os.path.join(input_data_path, 'anime.csv'))
    rating_df = pd.read_csv(os.path.join(input_data_path, 'rating.csv'))
    return anime_df, rating_df


# In[ ]:


def preprocess_data(anime_df):
    # 데이터 전처리를 수행합니다.
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(anime_df['genre'].fillna(''))
    return tfidf, tfidf_matrix  # tfidf 객체와 행렬을 반환합니다.


# In[ ]:

def calculate_cosine_similarity(tfidf_matrix):
    # 코사인 유사성 계산
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# 추천 생성 함수
def get_recommendations_excluding_negative_ratings(user_id, num_recommendations, all_ratings, anime_data, cosine_sim_matrix):
    user_ratings = all_ratings[all_ratings['user_id'] == user_id].copy()
    if user_ratings.empty:
        return pd.DataFrame(columns=['Anime', 'Similarity'])

    average_rating = all_ratings[all_ratings['rating'] != -1]['rating'].mean()
    user_ratings['rating'] = user_ratings['rating'].apply(lambda x: average_rating if x == -1 else x)

    # 표준편차가 0이 아닌 경우에만 z-score 계산을 수행합니다.
    if user_ratings['rating'].std() != 0:
        user_ratings['z_score'] = zscore(user_ratings['rating'])
        user_ratings = user_ratings[user_ratings['z_score'] > 0]
    else:
        # 표준편차가 0인 경우, 해당 사용자에 대한 추천은 생성하지 않습니다.
        return pd.DataFrame(columns=['Anime', 'Similarity'])

    recommendations = []
    for _, row in user_ratings.iterrows():
        anime_index = anime_data.index[anime_data['anime_id'] == row['anime_id']].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[anime_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        similar_animes = [{'Anime': anime_data['name'].iloc[i[0]], 'Similarity': i[1]} for i in sim_scores]
        recommendations.extend(similar_animes)

    # 추천 목록이 비어 있으면 비어 있는 DataFrame 반환
    if not recommendations:
        return pd.DataFrame(columns=['Anime', 'Similarity'])

    recommendations_df = pd.concat([pd.DataFrame([rec]) for rec in recommendations], ignore_index=True)
    return recommendations_df.drop_duplicates().sort_values('Similarity', ascending=False).head(num_recommendations)


# In[ ]:
if __name__ == '__main__':
    # SageMaker 환경에서는 학습 데이터가 이 경로에 저장됩니다.
    #input_data_path = '/opt/ml/input/data'  # SageMaker 환경의 데이터 경로
    output_model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    
    anime_df, rating_df = load_data()
    
    valid_ratings = rating_df[rating_df['rating'] != -1]
    #min_ratings = 5
    #valid_users = valid_ratings['user_id'].value_counts()[valid_ratings['user_id'].value_counts() >= min_ratings].index
    # 샘플링할 유저 비율 설정 (예: 전체 유저의 10%)
    sample_ratio = 0.001
    
    # 샘플링된 유저 ID 목록 생성
    sampled_users = valid_ratings['user_id'].drop_duplicates().sample(frac=sample_ratio)
    
    # 샘플링된 유저 데이터만 필터링
    sampled_data = valid_ratings[valid_ratings['user_id'].isin(sampled_users)]

    train_data_list = []
    test_data_list = []
    
    min_ratings = 2  # 최소한의 평가 데이터 수를 정합니다 (예: 2개)
    
    for user_id in sampled_data['user_id'].unique():
        user_ratings = sampled_data[sampled_data['user_id'] == user_id]
        
        # 사용자별 평가 데이터가 최소한의 수 이상인 경우에만 데이터 분할을 진행합니다.
        if len(user_ratings) >= min_ratings:
            train_data, test_data = train_test_split(user_ratings, test_size=0.2, random_state=42)
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        
    train_data = pd.concat(train_data_list)
    test_data = pd.concat(test_data_list)
    
    tfidf, tfidf_matrix = preprocess_data(anime_df)
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)
    
    precision_scores = []
    
    for user_id in test_data['user_id'].unique():
        actual_anime_ids = test_data[test_data['user_id'] == user_id]['anime_id'].tolist()
        user_recommendations = get_recommendations_excluding_negative_ratings(
            user_id,
            num_recommendations=5,
            all_ratings=train_data,
            anime_data=anime_df,
            cosine_sim_matrix=cosine_sim
        )
        recommended_anime_ids = user_recommendations['Anime'].tolist()
        user_precision = precision_at_k(recommended_anime_ids, actual_anime_ids, k=5)
        precision_scores.append(user_precision)
    
    mean_precision_at_5 = np.mean(precision_scores)
    print(f'Mean Precision@5: {mean_precision_at_5}')
    
    tfidf_filename = os.path.join(output_model_path, 'tfidf.joblib')
    joblib.dump(tfidf, tfidf_filename)

   
        
    #tfidf_filename = os.path.join(output_model_path, 'tfidf.joblib')
    #tfidf_matrix_filename = os.path.join(output_model_path, 'tfidf_matrix.joblib')
    #cosine_sim_filename = os.path.join(output_model_path, 'cosine_sim_matrix.joblib')
    
    #joblib.dump(tfidf, tfidf_filename)
    #joblib.dump(tfidf_matrix, tfidf_matrix_filename)
    #joblib.dump(cosine_sim, cosine_sim_filename)
    
