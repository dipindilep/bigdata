from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, when, from_unixtime

# Initialize Spark Session
spark = SparkSession.builder.appName("CleanAmazonReviews").getOrCreate()

# Read JSONL dataset
input_path = "gs://bigda25/Cell_Phones_and_Accessories.jsonl"
df = spark.read.json(input_path)

# Basic Cleaning
cleaned_df = df.select(
    col("rating").cast("float"),
    col("title"),
    col("text"),
    col("helpful_vote").cast("int"),
    col("verified_purchase").cast("boolean"),
    col("asin"),
    col("timestamp")
)

# Add derived features
final_df = cleaned_df.withColumn("review_length", length(col("text"))) \
                     .withColumn("review_date", from_unixtime(col("timestamp") / 1000).cast("timestamp")) \
                     .withColumn("is_helpful", when(col("helpful_vote") > 0, 1).otherwise(0))

# Save as Parquet
output_path = "gs://bigda25/cleaned_reviews_parquet"
final_df.write.mode("overwrite").parquet(output_path)
