import pandas as pd
import sqlalchemy

"""
Code Reference - https://teaching.mrsharky.com/sdsu_fall_2020_lecture04.html#/7/3/7
"""

"""
I was getting the error AttributeError: 'ObjectEngine' object has no attribute 'execute'
So I used SqlAlchemy < 2.0.0
Reference https://github.com/dagster-io/dagster/discussions/11881
"""


class BaseballDataLoader:
    # enter all the features here
    def __init__(self):
        self.dataset_name = "baseball"
        self.baseball_predictors = [
            "stadium_id",
            "pitchers_strikeout_to_walk_ratio_difference",
            "pitchers_opponent_batting_average_difference",
            "pitchers_strikeout_rate_difference",
            "pitchers_outsplayed_per_pitches_rate_difference",
            "pitchers_power_finesse_ratio_difference",
            "pitchers_whip_difference",
            "pitchers_dice_difference",
            "team_bat_avg_difference",
            "team_on_base_percentage_difference",
            "team_slugging_percentage_difference",
            "team_gross_production_average_difference",
            "team_walk_to_strikeout_ratio_difference",
            "temperature",
            "overcast",
            "wind",
            "winddir",
        ]

        self.baseball_response = "winner_home"

    def get_baseball_data(self):
        db_user = "root"  # change this to root when running in docker
        db_pass = "1998"  # pragma: allowlist secret
        db_host = "db_container"  # change this to db_container when running in docker
        db_database = "baseball"
        # pragma: allowlist secret
        connect_string = (
            f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
        )

        sql_engine = sqlalchemy.create_engine(connect_string)

        query = """
                SELECT * FROM fl_features_table
            """
        df = pd.read_sql_query(query, sql_engine)
        df_temp = pd.DataFrame()

        df_temp["stadium_id"] = df["stadium_id"]

        df_temp["pitchers_strikeout_to_walk_ratio_difference"] = (
            df["home_pitcher_strikeout_to_walk_ratio"]
            - df["away_pitcher_strikeout_to_walk_ratio"]
        )

        df_temp["pitchers_opponent_batting_average_difference"] = (
            df["home_pitcher_opponent_batting_average"]
            - df["away_pitcher_opponent_batting_average"]
        )

        df_temp["pitchers_strikeout_rate_difference"] = (
            df["home_pitcher_strikeout_rate"] - df["away_pitcher_strikeout_rate"]
        )

        df_temp["pitchers_outsplayed_per_pitches_rate_difference"] = (
            df["home_pitcher_outsplayed_per_pitches_rate"]
            - df["away_pitcher_outsplayed_per_pitches_rate"]
        )

        df_temp["pitchers_power_finesse_ratio_difference"] = (
            df["home_pitcher_power_finesse_ratio"]
            - df["away_pitcher_power_finesse_ratio"]
        )

        df_temp["pitchers_whip_difference"] = (
            df["home_pitcher_walks_hits_per_innings_pitched"]
            - df["away_pitcher_walks_hits_per_innings_pitched"]
        )

        df_temp["pitchers_dice_difference"] = (
            df["home_pitcher_dice"] - df["away_pitcher_dice"]
        )

        df_temp["team_bat_avg_difference"] = (
            df["home_team_bat_Avg"] - df["away_team_bat_Avg"]
        )

        df_temp["team_on_base_percentage_difference"] = (
            df["home_team_on_base_percentage"] - df["away_team_on_base_percentage"]
        )

        df_temp["team_slugging_percentage_difference"] = (
            df["home_team_slugging_percentage"] - df["away_team_slugging_percentage"]
        )

        df_temp["team_gross_production_average_difference"] = (
            1.8
            * (df["home_team_on_base_percentage"] - df["away_team_on_base_percentage"])
            + (
                df["home_team_slugging_percentage"]
                - df["away_team_slugging_percentage"]
            )
        ) / 4

        df_temp["team_walk_to_strikeout_ratio_difference"] = (
            df["home_team_walk_to_strikeout_ratio"]
            - df["away_team_walk_to_strikeout_ratio"]
        )

        df_temp["temperature"] = df["temperature"]
        df_temp["overcast"] = df["overcast"]
        df_temp["wind"] = df["wind"]
        df_temp["winddir"] = df["winddir"]

        df_temp["winner_home"] = df["winner_home"]
        # Filling all the NaN Values in each column with 0
        df_temp = df_temp.fillna(0)

        # Adding a column for game_date
        df_temp["game_date"] = df["game_date"]

        return (
            self.dataset_name,
            df_temp,
            self.baseball_predictors,
            self.baseball_response,
        )
