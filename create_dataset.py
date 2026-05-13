import pandas as pd
from sklearn.datasets import load_wine
import os


os.makedirs('data', exist_ok=True)

data = load_wine()

df = pd.DataFrame(data.data, columns= data.feature_names)
df['target'] = data.target

df.to_csv('data/wine_classification.csv', index=False) # Converting into CSV-file

print('csv file created')