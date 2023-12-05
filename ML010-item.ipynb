# Access to data from S3 by using Boto3
!pip install boto3

import boto3
#Get the file list from S3 bucket

s3 = boto3.client('s3')
bucket_name = 'gcu-ml02-010-item'  # Specify the S3 bucket name

# Get the list of objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Outputs the list of objects
for obj in response['Contents']:
    print(obj['Key'])

# load the data from S3

import pandas as pd

file_key1 = 'anime.csv'  # Specify the key of the file to be imported.
file_key2 = 'rating.csv'  # Specify the key of the file to be imported.

# Create the file URL of S3 bucket
file_url1 = f's3://{bucket_name}/{file_key1}'
file_url2 = f's3://{bucket_name}/{file_key2}'

# Use Pandas to load CSV files into DataFrame
anime = pd.read_csv(file_url1)
rating = pd.read_csv(file_url2)

# Preprocessing (select TV, Movie, OVA, ONA)
anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]

# Select only famous anime, 75% percentile
f = anime['members'].quantile(0.75)
anime = anime[(anime['members'] >= f)]

anime.head()

# Dataset Analysis
anime.isnull().sum()

# Replace missing rating with NaN
import numpy as np
rating.loc[rating.rating == -1, 'rating'] = np.NaN

rating.head()

rating.isnull().sum()

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

from sagemaker import get_execution_role

role = get_execution_role()

# Create SageMaker Estimator object
from sagemaker.sklearn.estimator import SKLearn

# Create Estimator object
estimator = SKLearn(
    entry_point='test.py', # File Name of Learning Script
    role=role, # role of SageMaker IAM
    instance_count=1, # Number of instances to use
    instance_type='ml.m5.large', # Instance Type
    framework_version='0.23-1' # Version of Scikit-learn
)

#Start Learning task
# Learning data path of S3 bucket
bucket_name = 'gcu-ml02-010-item'  # bucket name
data_path = ''  # There is no additional folder structure, so leave a blank

s3_input_train = f's3://{bucket_name}/{data_path}'

# Start Learning
estimator.fit({'train': s3_input_train})

# Model Deployment. Deploy the generated model to SageMaker end-point
endpoint_name = 'gcu-ml2-0104'
model_name = ''

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large", 
    endpoint_name=endpoint_name,
    model_name=model_name,
)
