import pyspark
import os
from dotenv import load_dotenv
from pathlib import Path

spark_host = "spark://dibimbing-dataeng-spark-master:7077"

dotenv_path = Path("/opt/app/.env")
load_dotenv(dotenv_path=dotenv_path)

postgres_host = os.getenv("DIBIMBING_DE_POSTGRES_HOST")
postgres_db = os.getenv("DIBIMBING_DE_POSTGRES_DB")
postgres_user = os.getenv("DIBIMBING_DE_POSTGRES_ACCOUNT")
postgres_password = os.getenv("DIBIMBING_DE_POSTGRES_PASSWORD")

kafka_host = os.getenv("DIBIMBING_DE_KAFKA_HOST")
kafka_topic = os.getenv("DIBIMBING_DE_KAFKA_TOPIC_NAME")

os.environ[
    "PYSPARK_SUBMIT_ARGS"
] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2 org.postgresql:postgresql:42.2.18"

sparkcontext = pyspark.SparkContext.getOrCreate(
    conf=(pyspark.SparkConf().setAppName("DibimbingStreaming").setMaster(spark_host))
)
sparkcontext.setLogLevel("WARN")
spark = pyspark.sql.SparkSession(sparkcontext.getOrCreate())

stream_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", f"{kafka_host}:9092")
    .option("subscribe", kafka_topic)
    .option("startingOffsets", "earliest")
    .load()
)

(
    stream_df.selectExpr("CAST(value AS STRING)")
    .writeStream.format("console")
    .outputMode("append")
    .start()
    .awaitTermination()
)
