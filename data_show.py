import pandas as pd

df = pd.read_parquet('gs://bigda25/cleaned_reviews_parquet/', engine='pyarrow', storage_options={'token': 'cloud'})
print(df.head())
