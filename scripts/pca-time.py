import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA
import json
import time

db = MongoClient('lattice-101', 27018)
print('log: connected to mongodb')

collection = 'noaa_nam'
print('collection:', collection)
cursor = db.sustaindb[collection].find()
df = pd.DataFrame(list(cursor))
print('log: dataframe created')

featuresFile = open('noaa_nam_features.json')

features = json.load(featuresFile)
print('log: features loaded')

total = df.shape[0]
print(f'total: {total}')

samples = [1.0, 0.5, 0.25, 0.1]

for sample in samples:
    print(f'sampling: {sample}')
    df = df.sample(frac=sample, replace=True, random_state=1)
    print(f'size: {df.shape[0]}')
    x = df.loc[:, features]
    x.loc[:] = normalize(x)
    start_time = time.monotonic()

    pca = PCA(n_components=len(features), random_state=2)
    x_pca = pca.fit_transform(x)

    print('Seconds:', (time.monotonic() - start_time))

    print() # new line 


