import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class FareTypePredictor:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.trained = False
        
    def train(self, fare_file='faretypes.csv'):
        """Train on historical fare data"""
        data = pd.read_csv(fare_file)
        # Features: origin_x, origin_y, dest_x, dest_y, time_of_day
        X = data[['originX', 'originY', 'destX', 'destY']].values
        y = data['FareType'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.trained = True
        print(f"Training accuracy: {self.model.score(X_test, y_test):.2f}")
        
    def predict_fare_type(self, origin, destination, call_time):
        """Predict fare type from origin/destination"""
        if not self.trained:
            return 'normal'  # default
            
        features = np.array([[origin[0], origin[1], 
                             destination[0], destination[1]]])
        return self.model.predict(features)[0]
