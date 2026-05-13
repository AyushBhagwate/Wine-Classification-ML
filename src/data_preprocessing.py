import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier


def get_pipeline():

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('yeo', PowerTransformer(method='yeo-johnson')),
        ('model', RandomForestClassifier(n_estimators=100))
    ])

    return model


    
