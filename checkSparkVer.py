from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("Test").getOrCreate()
print(spark.version)