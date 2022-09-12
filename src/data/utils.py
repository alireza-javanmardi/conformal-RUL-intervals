import pandas as pd
from sklearn.cluster import KMeans

class KMeansScaler:
    """A scaler that scale data based on the cluster they belong to 
    """
    def __init__(self, k: int, kmeans_features, base_scaler):
        self.k = k
        self.kmeans_features = kmeans_features
        self.kmeans_model = None

        #define a dictionary of transformers for each mode
        self.scalers = []

        for _ in range(k):
            self.scalers.append(base_scaler)

    def fit(self, df: pd.DataFrame):
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        self.kmeans_model = kmeans

        #cluster training
        modes = kmeans.fit_predict(df[self.kmeans_features])
        df_copy = df.copy()

        for i in range(self.k):
            self.scalers[i].fit(df_copy[modes == i])

    def transform(self, df: pd.DataFrame):
        assert self.kmeans_model is not None, "Scaler has to be fitted before transform."

        df_copy = df.copy()
        modes = self.kmeans_model.predict(df[self.kmeans_features])
        for i in range(self.k):
            df_copy[modes == i] = self.scalers[i].transform(df_copy[modes == i])
        df_copy["mode"] = modes
        return df_copy

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
