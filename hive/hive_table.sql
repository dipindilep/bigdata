CREATE EXTERNAL TABLE IF NOT EXISTS reviews_parquet (
    rating FLOAT,
    title STRING,
    text STRING,
    asin STRING,
    user_id STRING,
    timestamp BIGINT,
    helpful_vote INT,
    verified_purchase BOOLEAN,
    text_length INT,
    has_helpful_vote INT
)
STORED AS PARQUET
LOCATION 'gs://bigda25/processed_reviews_parquet/';
