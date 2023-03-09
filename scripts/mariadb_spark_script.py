import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

# Reference article: https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database

# Defining the global JDBC connection properties
database = "baseball"
user = "admin"
password = "1998"
server = "localhost"
port = 3306
jdbc_driver = "org.mariadb.jdbc.Driver"
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"

# Creating the spark session
spark = SparkSession.builder.appName(
    "Python Spark and Mariadb Connection"
).getOrCreate()


class RollingAverageTransformer(Transformer):
    def _transform(self, spark_int_bat_avg_df):
        # Generate new Schema
        spark_int_bat_avg_df.createOrReplaceTempView(
            "spark_intermediate_batting_average"
        )
        spark_int_bat_avg_df.persist(StorageLevel.DISK_ONLY)
        spark_100days_rolling_average = spark.sql(
            """
            SELECT ibba1.batter,
            COALESCE(SUM(ibba2.hit) / NULLIF(SUM(ibba2.atbat), 0), 0) AS bat_avg,
            DATE(ibba1.local_date) AS game_date
            FROM spark_intermediate_batting_average ibba1
            LEFT JOIN spark_intermediate_batting_average ibba2
            ON
            (((DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) <= 100)
            AND
            (DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) > 0))
            AND (ibba1.batter = ibba2.batter))
            GROUP BY ibba1.batter, DATE(ibba1.local_date)
            ORDER BY ibba1.batter, DATE(ibba1.local_date)
            """
        )

        return spark_100days_rolling_average


def return_spark_dataframe(sql):
    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("query", sql)
        .option("driver", jdbc_driver)
        .load()
    )
    return df


def main():
    sql = "SELECT * FROM im_batter_batting_average"
    df_im_table = return_spark_dataframe(sql)
    rolling_avg_transformer = RollingAverageTransformer()
    df_rolling_100_days_avg = rolling_avg_transformer.transform(df_im_table)
    df_rolling_100_days_avg.createOrReplaceTempView("spark_rolling_100_days_avg")
    df_rolling_100_days_avg.persist(StorageLevel.DISK_ONLY)

    # Displaying the final results for one batter 110029
    results = spark.sql(
        """
            SELECT batter, bat_avg, game_date FROM spark_rolling_100_days_avg
            WHERE batter = 110029
        """
    )

    results.show()


if __name__ == "__main__":
    sys.exit(main())
