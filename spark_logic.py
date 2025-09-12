import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
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

    os.environ["PYSPARK_PYTHON"] = r"C:\Users\muham\Desktop\StockPulse\venv311\Scripts\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\muham\Desktop\StockPulse\venv311\Scripts\python.exe"

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
    
    pdf = pdf.dropna().reset_index(drop=True)
    
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

def detect_anomalies_from_transformed(df_trans: DataFrame, k=3, quantile=0.99, seed=42):

    # 1. Train KMeans
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=seed)
    kmeans_model = kmeans.fit(df_trans)

    # 2. Assign clusters
    df_clustered = kmeans_model.transform(df_trans)

    # 3. Distance from cluster centers
    centers = kmeans_model.clusterCenters()

    def distance_to_center(features, cluster):
        center = centers[cluster]
        return float(np.linalg.norm(features - center))

    distance_udf = F.udf(distance_to_center, DoubleType())
    df_clustered = df_clustered.withColumn(
        "distance", distance_udf(F.col("features"), F.col("cluster"))
    )

    # 4. Detect anomalies by distance threshold
    threshold = df_clustered.approxQuantile("distance", [quantile], 0.01)[0]
    df_anomalies = df_clustered.filter(F.col("distance") > threshold)

    print(f"Anomaly distance threshold: {threshold}")

    return df_clustered, df_anomalies, centers, threshold