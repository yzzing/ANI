#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore
import joblib


# In[ ]:


# SageMaker 환경에서는 학습 데이터가 이 경로에 저장됩니다.
input_data_path = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')

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


# In[ ]:


def get_recommendations_for_user(user_id, num_recommendations, all_ratings, anime_data, cosine_sim_matrix):
    # 사용자가 평가한 애니메이션 가져오기
    user_ratings = all_ratings[all_ratings['user_id'] == user_id].copy()
    
    # 모든 평가가 -1인 경우를 확인하고 처리
    if user_ratings['rating'].eq(-1).all():
        # 전체 데이터셋의 평균을 사용하거나 기본값 설정
        average_rating = all_ratings[all_ratings['rating'] != -1]['rating'].mean()
        user_ratings['rating'] = average_rating
    else:
        # 평가하지 않은 항목(-1)에 평균 점수 할당
        user_ratings.loc[user_ratings['rating'] == -1, 'rating'] = all_ratings[all_ratings['rating'] != -1]['rating'].mean()
    
    # 사용자의 평균과 표준편차를 기반으로 Z-점수 계산
    # 표준편차가 0이면 (즉, 모든 평가가 동일하면) Z-점수 계산은 NaN을 반환하므로, 이를 처리
    if user_ratings['rating'].std() > 0:
        user_ratings['z_score'] = zscore(user_ratings['rating'])
        user_ratings = user_ratings[user_ratings['z_score'] > 0]

    # 유사한 애니메이션 찾기
    recommendations = []
    for _, row in user_ratings.iterrows():
        anime_index = anime_data.index[anime_data['anime_id'] == row['anime_id']].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[anime_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        similar_animes = [{'Anime': anime_data['name'].iloc[i[0]], 'Similarity': i[1]} for i in sim_scores]
        recommendations.extend(similar_animes)

    # 리스트를 DataFrame으로 변환
    recommendations_df = pd.concat([pd.DataFrame([rec]) for rec in recommendations], ignore_index=True)

    # 중복 제거 및 유사성에 따른 정렬
    recommendations_df = recommendations_df.drop_duplicates().sort_values('Similarity', ascending=False).head(num_recommendations)
    
    return recommendations_df


# In[ ]:


# 2. 가중치 부여 (시청은 했지만 평가하지 않은 항목에 낮은 가중치 부여)

def get_recommendations_for_user_with_weights(user_id, num_recommendations, all_ratings, anime_data, cosine_sim_matrix):
    # 사용자가 평가한 애니메이션 가져오기
    user_ratings = all_ratings[all_ratings['user_id'] == user_id].copy()

    # 평가하지 않은 항목(-1)에 낮은 가중치 부여
    user_ratings['weight'] = user_ratings['rating'].apply(lambda x: 0.5 if x == -1 else 1)

    # 유사한 애니메이션 찾기
    all_recommendations = []  # 추천들을 저장할 리스트
    for _, row in user_ratings.iterrows():
        anime_index = anime_data.index[anime_data['anime_id'] == row['anime_id']].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[anime_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]  # 자기 자신을 제외
        similar_animes = [(anime_data['name'].iloc[i[0]], i[1], i[1] * row['weight']) for i in sim_scores]
        all_recommendations.extend(similar_animes)

    # 리스트를 DataFrame으로 변환
    recommendations = pd.DataFrame(all_recommendations, columns=['Anime', 'Similarity', 'Weighted Similarity'])

    # 가중치에 따른 유사성을 기준으로 중복 제거 및 정렬
    recommendations = recommendations.drop_duplicates(subset=['Anime']).sort_values('Weighted Similarity', ascending=False).head(num_recommendations)
    return recommendations


# In[ ]:


# 3. 이진 분류 (시청 여부만 고려)
def get_recommendations_for_user_binary_classification(user_id, num_recommendations, all_ratings, anime_data, cosine_sim_matrix):
    # 사용자가 평가한 애니메이션 가져오기
    user_ratings = all_ratings[all_ratings['user_id'] == user_id].copy()

    # 시청 여부를 이진 값으로 변환
    user_ratings['watched'] = user_ratings['rating'].apply(lambda x: 1 if x != -1 else 0)

    # 시청한 애니메이션만 고려
    watched_animes = user_ratings[user_ratings['watched'] == 1]

    # 유사한 애니메이션 찾기
    recommendations = []
    for _, row in watched_animes.iterrows():
        anime_index = anime_data.index[anime_data['anime_id'] == row['anime_id']].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[anime_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]  # 자기 자신을 제외
        similar_animes = [{'Anime': anime_data['name'].iloc[i[0]], 'Similarity': i[1]} for i in sim_scores]
        recommendations.extend(similar_animes)

    # 리스트를 DataFrame으로 변환
    recommendations_df = pd.DataFrame(recommendations, columns=['Anime', 'Similarity'])

    # 중복 제거 및 유사성에 따른 정렬
    recommendations_df = recommendations_df.drop_duplicates().sort_values('Similarity', ascending=False).head(num_recommendations)
    return recommendations_df


# In[ ]:


# -1 평가 제거 후 추천
def get_recommendations_excluding_negative_ratings(user_id, num_recommendations, all_ratings, anime_data, cosine_sim_matrix):
    # 사용자가 평가한 애니메이션 가져오기 (평가 -1 제외)
    user_ratings = all_ratings[(all_ratings['user_id'] == user_id) & (all_ratings['rating'] != -1)].copy()

    # 유사한 애니메이션 찾기
    recommendations = []
    for _, row in user_ratings.iterrows():
        anime_index = anime_data.index[anime_data['anime_id'] == row['anime_id']].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[anime_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]  # 자기 자신을 제외
        similar_animes = [{'Anime': anime_data['name'].iloc[i[0]], 'Similarity': i[1]} for i in sim_scores]
        recommendations.extend(similar_animes)

    # 리스트를 DataFrame으로 변환
    recommendations_df = pd.DataFrame(recommendations, columns=['Anime', 'Similarity'])

    # 중복 제거 및 유사성에 따른 정렬
    recommendations_df = recommendations_df.drop_duplicates().sort_values('Similarity', ascending=False).head(num_recommendations)
    return recommendations_df


# 모델 로드 함수
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

# 입력 데이터 처리 함수
def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        # 다른 컨텐트 타입에 대한 처리를 추가할 수 있습니다.
        raise ValueError(f'Unsupported content type: {request_content_type}')

# 예측 함수
def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions

# 출력 데이터 처리 함수
def output_fn(prediction, content_type):
    if content_type == 'text/csv':
        return pd.DataFrame(prediction).to_csv(index=False)
    else:
        # 다른 컨텐트 타입에 대한 처리를 추가할 수 있습니다.
        raise ValueError(f'Unsupported content type: {content_type}')


# In[ ]:

if __name__ == '__main__':
    anime_df, rating_df = load_data()
    tfidf, tfidf_matrix = preprocess_data(anime_df)  
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)
    
    #추천 생성 함수 호출.
    print("<get_recommendations_excluding_negative_ratings , user_id=1>")
    recommendations = get_recommendations_excluding_negative_ratings(
        user_id=1,
        num_recommendations=10,
        all_ratings=rating_df,
        anime_data=anime_df,
        cosine_sim_matrix=cosine_sim
    )
    print(recommendations)
    
    # 모델과 유사성 행렬 저장
    output_model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    tfidf_matrix_filename = os.path.join(output_model_path, 'tfidf_matrix.joblib')
    cosine_sim_filename = os.path.join(output_model_path, 'cosine_sim_matrix.joblib')
    
    # tfidf 객체 저장
    tfidf_filename = os.path.join(output_model_path, 'tfidf.joblib') 
    joblib.dump(tfidf, tfidf_filename)  
    

    joblib.dump(tfidf, tfidf_matrix_filename)  # tfidf 객체를 저장
    joblib.dump(cosine_sim, cosine_sim_filename)  # 코사인 유사성 행렬을 저장

