#Boto3를 사용하여 S3에서 데이터 접근하기
!pip install boto3

import boto3

#S3 버킷에서 파일 리스트 가져오기

s3 = boto3.client('s3')
bucket_name = 'gcu-ml02-010'  # S3 버킷 이름을 지정합니다.

# 버킷 내 객체 목록을 가져옵니다.
response = s3.list_objects_v2(Bucket=bucket_name)

# 객체 목록을 출력합니다.
for obj in response['Contents']:
    print(obj['Key'])


#S3에서 데이터 로드하기

import pandas as pd

file_key1 = 'anime.csv'  # 가져올 파일의 키를 지정합니다.
file_key2 = 'rating.csv'  # 가져올 파일의 키를 지정합니다.

# S3 버킷의 파일 URL을 생성합니다.
file_url1 = f's3://{bucket_name}/{file_key1}'
file_url2 = f's3://{bucket_name}/{file_key2}'

# Pandas를 사용하여 CSV 파일을 DataFrame으로 로드합니다.
anime_df = pd.read_csv(file_url1)
rating_df = pd.read_csv(file_url2)

anime_df.head()

#데이터셋 분석
anime_df.isnull().sum()

rating_df.head()

rating_df.isnull().sum()

import matplotlib.pyplot as plt

# 쉼표로 구분된 장르들을 분할
genres = anime_df['genre'].str.split(', ', expand=True)
# genres 데이터프레임에서 장르별로 나타난 빈도수를 카운트
genre_counts = pd.Series(genres.values.ravel()).value_counts()

# 장르별 카운트를 바 플롯으로 표시
plt.figure(figsize=(10,8))
genre_counts.plot(kind='bar')
plt.title('animation cont')
plt.xlabel('genre')
plt.ylabel('Num')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 'rating' 칼럼의 내용 분포를 계산합니다.
rating_distribution = rating_df['rating'].value_counts().sort_index()

# 분포를 DataFrame으로 변환하고 출력합니다.
rating_distribution_df = pd.DataFrame(rating_distribution)
rating_distribution_df.reset_index(inplace=True)
rating_distribution_df.columns = ['Rating', 'Frequency']

# 분포를 막대 그래프로 시각화합니다.
plt.figure(figsize=(10, 5))
plt.bar(rating_distribution_df['Rating'], rating_distribution_df['Frequency'], color='skyblue')
plt.title('Rating Distribution in rating.csv')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(-1, 11))  # -1부터 10까지의 평점을 표시
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

from sagemaker import get_execution_role

role = get_execution_role()

#SageMaker Estimator 생성
from sagemaker.sklearn.estimator import SKLearn

# Estimator 객체 생성
estimator = SKLearn(
    entry_point='train_script.py', # 학습 스크립트 파일명
    role=role, # SageMaker IAM 역할
    instance_count=1, # 사용할 인스턴스 개수
    instance_type='ml.m5.large', # 인스턴스 유형
    framework_version='0.23-1' # Scikit-learn 버전
)

#학습 작업 시작
# S3 버킷의 학습 데이터 경로
bucket_name = 'gcu-ml02-010'  # 버킷 이름
data_path = '' 

s3_input_train = f's3://{bucket_name}/{data_path}'

# 학습 작업 시작
estimator.fit({'train': s3_input_train})

# 모델 아티팩트의 경로 가져오기
model_path = estimator.model_data
print(model_path)

from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model import Model

model_data=f's3://{bucket_name}/{model_path}/model.tar.gz'


model = SKLearnModel(
    model_data=model_data,  # 모델 아티팩트의 S3 경로
    role=role,  
    entry_point='trainTest_script.py',
    framework_version='0.23-1',  # 사용하고 싶은 Scikit-learn 버전
    py_version='py3', 
)

#모델 배포.  생성된 모델을 SageMaker 엔드포인트로 배포
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')





