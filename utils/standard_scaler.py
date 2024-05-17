class StandardScaler:
    def fit_transform(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std