import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.linalg import Vectors
import os
import sys
import numpy as np

def load_csv_with_fix(path, spark):
    pdf = pd.read_csv(path, skiprows=2)
    pdf.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")

    return spark.createDataFrame(pdf)

def create_spark_session() -> SparkSession:

    venv_python = r"C:\Users\muham\Desktop\StockPulse-Latest\venv\Scripts\python.exe"

    os.environ["PYSPARK_PYTHON"] = venv_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    spark = SparkSession.builder \
    .appName("StockPulse") \
    .config("spark.python.worker.faulthandler.enabled", "true") \
    .master("local[1]") \
    .getOrCreate()

    return spark

def engineer_features(spark: SparkSession, initial_df: DataFrame) -> DataFrame:
    print("Columns in Spark DF:", initial_df.columns)

    pdf = initial_df.toPandas()
    
    # Feature Engineering Logic
    pdf['Date'] = pd.to_datetime(pdf['Date'])
    pdf = pdf.set_index('Date').sort_index()

    pdf['Return'] = pdf['Close'].pct_change().fillna(0)

    pdf['MA_7'] = pdf['Close'].rolling(window=7).mean()
    pdf['MA_30'] = pdf['Close'].rolling(window=30).mean()
    pdf['Vol_14'] = pdf['Return'].rolling(window=14).std()

    delta = pdf['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    pdf['RSI_14'] = 100 - (100 / (1 + rs))

    ema_26 = pdf['Close'].ewm(span=26, adjust=False).mean()
    ema_12 = pdf['Close'].ewm(span=12, adjust=False).mean()
    pdf['MACD'] = ema_12 - ema_26
    pdf['MACD_Signal'] = pdf['MACD'].ewm(span=9, adjust=False).mean()
    
    pdf = pdf.dropna().reset_index()
    
    # Convert back to a Spark DataFrame
    return spark.createDataFrame(pdf)

def transform_features(df: DataFrame) -> DataFrame:
    
    feature_cols = [
        "Close", "High", "Low", "Open", "Volume",
        "Return", "MA_7", "MA_30", "Vol_14", "RSI_14", "MACD", "MACD_Signal"
    ]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_assembled",
        handleInvalid="keep"
    )

    scaler = StandardScaler(
        inputCol="features_assembled",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    pipeline = Pipeline(stages=[assembler, scaler])
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)
    
    print("Transformed schema:")
    df_transformed.select("features").printSchema()
    
    return df_transformed

def detect_anomalies_KMeans(df_trans: DataFrame, k=5, seed=42, threshold_std_dev=3.0):
    
    # 1. Train KMeans (No change)
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=seed)
    kmeans_model = kmeans.fit(df_trans)

    # 2. Assign clusters (No change)
    df_clustered = kmeans_model.transform(df_trans)

    # 3. Calculate distance from cluster centers (No change)
    centers = kmeans_model.clusterCenters()

    def distance_to_center(features, cluster):
        center = centers[cluster]
        return float(np.linalg.norm(features - center))

    distance_udf = F.udf(distance_to_center, DoubleType())
    df_clustered = df_clustered.withColumn(
        "distance", distance_udf(F.col("features"), F.col("cluster"))
    )

    dist_stats = df_clustered.select(
        F.mean("distance").alias("mean_dist"),
        F.stddev("distance").alias("std_dev_dist")
    ).collect()[0]

    mean_dist = dist_stats["mean_dist"]
    std_dev_dist = dist_stats["std_dev_dist"]
    
    threshold = mean_dist + threshold_std_dev * std_dev_dist

    print(f"Mean distance: {mean_dist:.4f}, StdDev distance: {std_dev_dist:.4f}")
    print(f"Anomaly distance threshold set at {threshold:.4f}")

    # 5. Filter for anomalies using the new threshold (No change in logic)
    df_anomalies = df_clustered.filter(F.col("distance") > threshold)

    return df_clustered, df_anomalies, centers, threshold

def detect_anomalies_GMM(transformed_df, k=5, seed=1, threshold_std_dev=3.0):
    """
    Detects anomalies using a Gaussian Mixture Model.
    """
    print(f"Running GMM with k={k} and threshold multiplier={threshold_std_dev}")
    gmm = GaussianMixture(featuresCol='features', k=k, seed=seed)
    model = gmm.fit(transformed_df)
    predictions = model.transform(transformed_df)

    # UDF to get the max probability from the probability vector output by GMM
    max_prob_udf = udf(lambda v: float(max(v)), DoubleType())
    
    # Anomaly score: Negative log of the maximum probability. Higher score = more anomalous.
    # We add a small epsilon to prevent log(0) errors.
    predictions_with_score = predictions.withColumn("max_prob", max_prob_udf(col("probability"))) \
                                        .withColumn("distance", -F.log(col("max_prob") + 1e-9)) # <-- FIX: Changed spark_log to F.log

    # Calculate the threshold based on the mean and standard deviation of the score
    score_stats = predictions_with_score.select(
        F.mean(col("distance")).alias("mean_score"),     # <-- FIX: Added F. prefix
        F.stddev(col("distance")).alias("stddev_score") # <-- FIX: Added F. prefix
    ).collect()[0]

    mean_score = score_stats["mean_score"]
    stddev_score = score_stats["stddev_score"]
    threshold = mean_score + threshold_std_dev * stddev_score

    anomalies_df = predictions_with_score.filter(col("distance") > threshold)
    df_clustered = predictions_with_score.filter(col("distance") <= threshold)

    # For GMM, "centers" are the means of each Gaussian component
    centers = [g.mean.toArray().tolist() for g in model.gaussiansDF.select("mean").collect()]

    return df_clustered, anomalies_df, centers, threshold