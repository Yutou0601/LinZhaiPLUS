# model/knn.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_similar_listings(listings, target_listing, n_neighbors=3):
    """
    使用 KNN 找出最相仿的租屋資訊（只推薦相同城市的房源）
    :param listings: 該市區的所有租屋資訊 (list of dicts)
    :param target_listing: 當前瀏覽的租屋資訊 (dict)
    :param n_neighbors: 取最相仿的數量
    :return: 最相仿的租屋資訊列表
    """
  
    city = target_listing['city']
    listings_same_city = [listing for listing in listings if listing['city'] == city]

    if len(listings_same_city) < 1:
        return []   

    n_neighbors = min(n_neighbors, len(listings_same_city), 3)  

    features = ['price', 'bedroom', 'age', 'floors']  
    data_matrix = np.array([[listing[feature] for feature in features] for listing in listings_same_city])
    target_features = np.array([target_listing[feature] for feature in features]).reshape(1, -1)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(data_matrix)
    distances, indices = knn.kneighbors(target_features)

    similar_listings = [listings_same_city[idx] for idx in indices[0]]
    
    return similar_listings
