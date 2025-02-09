import sys

from pyspark.sql import SparkSession


def main():
    # Initialize Spark Session
    spark = SparkSession.builder.appName("ParquetToCSV").getOrCreate()
    
    # Read arguments passed by EMR Step
    args = sys.argv
    input_path = args[args.index("--input") + 1]
    output_path = args[args.index("--output") + 1]

    # Read Parquet file from S3
    df = spark.read.parquet(input_path)

    # Take a sample of 10,000 rows
    sampled_df = df.limit(10000)

    # Save as a single CSV file (coalesce(1) forces single output file)
    sampled_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    # Stop Spark
    spark.stop()

if __name__ == "__main__":
    main()
