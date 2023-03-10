import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

# Reference article: https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database

# Defining the global JDBC connection properties
database = "baseball"
server = "localhost"
port = 3306
jdbc_driver = "org.mariadb.jdbc.Driver"
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"


# Rolling batting average Transformer class
class RollingAverageTransformer(Transformer):
    def _transform(self, spark, df_intermediate_batting_avg):
        """
        This function uses spark dataframe as input, performs the transformation and returns the results of the query
        as spark dataframe.
        :param df_intermediate_batting_avg: spark dataframe
        :return: spark_100days_rolling_average: spark dataframe
        """
        df_intermediate_batting_avg.createOrReplaceTempView(
            "spark_intermediate_batting_average"
        )
        df_intermediate_batting_avg.persist(StorageLevel.DISK_ONLY)
        spark_100days_rolling_average = spark.sql(
            """
            SELECT ibba1.batter,
            ROUND(COALESCE(SUM(ibba2.hit) / NULLIF(SUM(ibba2.atbat), 0), 0),4) AS bat_avg,
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


def fetch_mariadb_data(spark, sql, user, password):
    """
    Function takes Database credentials and SQL query and returns the spark dataframe of the output
    :param spark: spark session
    :param sql: string
    :param user: string
    :param password: string
    :return: spark dataframe
    """
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


def get_intermediate_table(spark, df_batter_counts, df_game):
    """
    Function takes spark dataframes, makes their temp view and runs the SQL on spark to create intermediate temp table.
    Returns the results in the form of spark dataframe.
    :param spark: spark session
    :param df_batter_counts:
    :param df_game:
    :return: df_intermediate_batting_avg:
    """
    df_batter_counts.createOrReplaceTempView("batter_counts")
    df_batter_counts.persist(StorageLevel.DISK_ONLY)

    df_game.createOrReplaceTempView("game")
    df_game.persist(StorageLevel.DISK_ONLY)

    df_intermediate_batting_avg = spark.sql(
        """
            SELECT bc.game_id AS game_id,
            bc.batter AS batter,
            bc.atbat AS atbat,
            bc.hit AS hit,
            g.local_date AS local_date,
            g.et_date AS et_date
            FROM batter_counts bc
            INNER JOIN game g ON bc.game_id = g.game_id
            ORDER BY local_date
        """
    )

    return df_intermediate_batting_avg


def main():
    # Mariadb Username and Password taking input during runtime
    user = input("Enter Mariadb Username: ")
    password = input("Enter Mariadb Password: ")

    try:
        # Creating the spark session
        spark = SparkSession.builder.appName(
            "Python Spark and Mariadb Connection"
        ).getOrCreate()

        # Getting mariadb table batter_counts into spark dataframe
        sql_batter_counts = "SELECT * FROM batter_counts"
        df_batter_counts = fetch_mariadb_data(spark, sql_batter_counts, user, password)

        # Getting mariadb table game into spark dataframe
        sql_game = "SELECT * FROM game"
        df_game = fetch_mariadb_data(spark, sql_game, user, password)

    except Exception as e:
        print(f"Error in fetching data from mariadb: {e}")

    # Getting the intermediate table by joining batter_counts and game table in spark
    df_intermediate_batting_avg = get_intermediate_table(
        spark, df_batter_counts, df_game
    )

    rolling_100_days_avg_table = "spark_rolling_100_days_avg"

    # Calculating 100 days rolling batting average in spark transformer
    rolling_avg_transformer = RollingAverageTransformer()
    df_rolling_100_days_avg = rolling_avg_transformer._transform(
        spark, df_intermediate_batting_avg
    )

    # Creating the temp view for 100 days rolling batting average results
    df_rolling_100_days_avg.createOrReplaceTempView(rolling_100_days_avg_table)
    df_rolling_100_days_avg.persist(StorageLevel.DISK_ONLY)

    # Displaying the final results for one batter 110029. Please change the query to test for other batters
    results = spark.sql(
        """
            SELECT batter, bat_avg, game_date FROM spark_rolling_100_days_avg
            WHERE batter = 110029
        """
    )

    results.show()


if __name__ == "__main__":
    sys.exit(main())
