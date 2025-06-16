# Spark ML Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, rand
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Python + ML Imports 
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

# 1. Start Spark
spark = SparkSession.builder.appName("RealisticHelpfulnessPrediction").getOrCreate()

# 2. Load data
df = spark.read.parquet("gs://bigda25/cleaned_reviews_parquet")

# 3. Clean + label
df_clean = df.dropna(subset=["rating", "title", "text"]).dropDuplicates(["title", "text"])
df_labeled = df_clean.withColumn("is_helpful", when(col("rating") >= 4, 1).otherwise(0))

# 4. Balance classes
df_pos = df_labeled.filter(col("is_helpful") == 1).sample(0.3, seed=1)
df_neg = df_labeled.filter(col("is_helpful") == 0).sample(0.3, seed=1)
df_balanced = df_pos.union(df_neg).orderBy(rand())

# 5. Combine title + text
df_final = df_balanced.withColumn("combined_text", concat_ws(" ", col("title"), col("text")))

# 6. NLP + ML pipeline
tokenizer = Tokenizer(inputCol="combined_text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(labelCol="is_helpful", featuresCol="features")
pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, lr])

# 7. Train-test split
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

# 8. Train model
model = pipeline.fit(train_data)

# 9. Predict
predictions = model.transform(test_data)

# 10. Evaluate with Spark
evaluator = BinaryClassificationEvaluator(labelCol="is_helpful", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"\n ROC AUC Score: {roc_auc:.4f}")

# 11. Evaluate with sklearn
pdf = predictions.select("is_helpful", "prediction").sample(False, 0.1, seed=42).limit(5000).toPandas()
y_true = pdf["is_helpful"]
y_pred = pdf["prediction"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
report = classification_report(y_true, y_pred)

print("\n Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nFull Classification Report:\n", report)

# 12. Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
labels = ["Not Helpful", "Helpful"]

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save and upload
local_path = "confusion_matrix.png"
plt.savefig(local_path)
plt.close()

subprocess.run(["gsutil", "cp", local_path, "gs://bigda25/confusion_matrix.png"], check=True)
print("Confusion matrix uploaded to: gs://bigda25/confusion_matrix.png")
