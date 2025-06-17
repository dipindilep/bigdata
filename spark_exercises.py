from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, date_format
import time

spark = SparkSession.builder.appName("SparkExerciseTasks").getOrCreate()

# Load parquet data
df = spark.read.parquet("gs://bigda25/cleaned_reviews_parquet")

base_output_path = "gs://bigda25/spark_exercise_output/"

# avg_monthly_rating
start = time.time()
avg_monthly_rating = (
    df.withColumn("month", date_format(col("review_date"), "yyyy-MM"))
      .groupBy("month")
      .agg(avg("rating").alias("avg_rating"))
      .orderBy("month")
)
avg_monthly_rating.show(10, truncate=False)
avg_monthly_rating.write.mode("overwrite").parquet(base_output_path + "avg_monthly_rating/")
end = time.time()
print(f"Task 1 (avg_monthly_rating) completed in {end - start:.2f} seconds")

# top_reviewed_products (top 10)
start = time.time()
top_reviewed_products = (
    df.groupBy("asin")
      .agg(count("*").alias("review_count"))
      .orderBy(col("review_count").desc())
      .limit(10)
)
top_reviewed_products.show(truncate=False)
top_reviewed_products.write.mode("overwrite").parquet(base_output_path + "top_reviewed_products/")
end = time.time()
print(f"Task 2 (top_reviewed_products) completed in {end - start:.2f} seconds")

# verified_vs_unverified
start = time.time()
verified_vs_unverified = (
    df.groupBy("verified_purchase")
      .agg(
          count("*").alias("review_count"),
          avg("rating").alias("avg_rating")
      )
)
verified_vs_unverified.show(truncate=False)
verified_vs_unverified.write.mode("overwrite").parquet(base_output_path + "verified_vs_unverified/")
end = time.time()
print(f"Task 3 (verified_vs_unverified) completed in {end - start:.2f} seconds")

spark.stop()
