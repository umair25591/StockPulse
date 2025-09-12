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
from pyspark.ml.stat import Correlation


def load_csv_with_fix(path, spark):
    pdf = pd.read_csv(path)
    rename_map = {
        'close': 'Close', 'high': 'High', 'low': 'Low', 
        'open': 'Open', 'volume': 'Volume', 'date': 'Date'
    }
    pdf.columns = [col.lower() for col in pdf.columns]
    pdf.rename(columns=rename_map, inplace=True)
    
    pdf.columns = [col.capitalize() for col in pdf.columns]

    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")
    return spark.createDataFrame(pdf.dropna(subset=['Date']))

def create_spark_session() -> SparkSession:

    venv_python = r"C:\Users\muham\Desktop\StockPulse\venv311\Scripts\python.exe"

    os.environ["PYSPARK_PYTHON"] = venv_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    spark = SparkSession.builder \
    .appName("StockPulse") \
    .config("spark.python.worker.faulthandler.enabled", "true") \
    .master("local[1]") \
    .getOrCreate()

    return spark

def engineer_features(spark: SparkSession, initial_df: DataFrame) -> DataFrame:
    pdf = initial_df.toPandas()
    
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
    
    return spark.createDataFrame(pdf)

def transform_features(df: DataFrame, selected_features: list) -> DataFrame:
    assembler = VectorAssembler(
        inputCols=selected_features,
        outputCol="features_assembled",
        handleInvalid="error"
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
    
    return df_transformed

def select_features(df: DataFrame, correlation_threshold=0.95) -> list:
    all_feature_cols = [
        "Close", "High", "Low", "Open", "Volume", "Return", 
        "MA_7", "MA_30", "Vol_14", "RSI_14", "MACD", "MACD_Signal"
    ]
    
    assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="corr_features")
    df_vector = assembler.transform(df).select("corr_features")

    matrix = Correlation.corr(df_vector, "corr_features").head()
    corr_matrix = matrix[0].toArray()

    print("\n--- Feature Selection: Correlation Matrix ---")
    print(pd.DataFrame(corr_matrix, columns=all_feature_cols, index=all_feature_cols))
    
    columns_to_drop = set()
    for i in range(len(all_feature_cols)):
        for j in range(i + 1, len(all_feature_cols)):
            if abs(corr_matrix[i, j]) >= correlation_threshold:
                columns_to_drop.add(all_feature_cols[j])

    if columns_to_drop:
        print(f"\nDropping highly correlated features: {list(columns_to_drop)}")
    
    final_features = [col for col in all_feature_cols if col not in columns_to_drop]
    
    print(f"Selected features for model: {final_features}\n")
    
    return final_features

def detect_anomalies_KMeans(df_trans: DataFrame, k=5, seed=42, threshold_std_dev=3.0): 
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=seed)
    kmeans_model = kmeans.fit(df_trans)

    df_clustered = kmeans_model.transform(df_trans)

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

    df_anomalies = df_clustered.filter(F.col("distance") > threshold)

    return df_clustered, df_anomalies, centers, threshold

def detect_anomalies_GMM(transformed_df, k=5, seed=1, threshold_std_dev=3.0):
    gmm = GaussianMixture(featuresCol='features', k=k, seed=seed)
    model = gmm.fit(transformed_df)
    predictions = model.transform(transformed_df)

    max_prob_udf = udf(lambda v: float(max(v)), DoubleType())
    
    predictions_with_score = predictions.withColumn("max_prob", max_prob_udf(col("probability"))) \
                                        .withColumn("distance", -F.log(col("max_prob") + 1e-9))

    score_stats = predictions_with_score.select(
        F.mean(col("distance")).alias("mean_score"),
        F.stddev(col("distance")).alias("stddev_score")
    ).collect()[0]

    mean_score = score_stats["mean_score"]
    stddev_score = score_stats["stddev_score"]
    threshold = mean_score + threshold_std_dev * stddev_score

    anomalies_df = predictions_with_score.filter(col("distance") > threshold)
    df_clustered = predictions_with_score.filter(col("distance") <= threshold)

    centers = [g.mean.toArray().tolist() for g in model.gaussiansDF.select("mean").collect()]

    return df_clustered, anomalies_df, centers, threshold



def load_csv_pandas(path):
    pdf = pd.read_csv(path)
    rename_map = {
        'close': 'Close', 'high': 'High', 'low': 'Low', 
        'open': 'Open', 'volume': 'Volume', 'date': 'Date'
    }
    pdf.columns = [col.lower() for col in pdf.columns]
    pdf.rename(columns=rename_map, inplace=True)
    
    pdf.columns = [col.capitalize() for col in pdf.columns]

    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")

    return pdf

def engineer_features_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf['Date'] = pd.to_datetime(pdf['Date'])
    pdf = pdf.set_index('Date').sort_index()

    pdf['Return'] = pdf['Close'].pct_change()
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
    
    return pdf.dropna().reset_index()

def select_features_pandas(pdf: pd.DataFrame, correlation_threshold=0.95) -> list:
    all_feature_cols = [
        "Close", "High", "Low", "Open", "Volume", "Return", 
        "MA_7", "MA_30", "Vol_14", "RSI_14", "MACD", "MACD_Signal"
    ]
    
    corr_matrix = pdf[all_feature_cols].corr()

    columns_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= correlation_threshold:
                colname = corr_matrix.columns[i]
                columns_to_drop.add(colname)
    
    if columns_to_drop:
        print(f"\nDropping highly correlated features: {list(columns_to_drop)}")
        
    final_features = [col for col in all_feature_cols if col not in columns_to_drop]
    print(f"Selected features for model: {final_features}\n")
    
    return final_features