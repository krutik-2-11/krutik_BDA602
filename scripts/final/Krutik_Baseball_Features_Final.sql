USE baseball;

-- Should compare these features between home team and away team

-- Combining game and box_score table
DROP TABLE IF EXISTS im_game_table;
CREATE TABLE im_game_table AS
SELECT
	g.game_id
	, DATE(g.local_date) AS game_date
	, g.home_pitcher
	, g.home_team_id
	, g.away_pitcher
	, g.away_team_id
	, CAST(REPLACE(SUBSTRING_INDEX(bs.temp, ' ', 1), ' degrees', '') AS DECIMAL(10, 2)) AS temperature
	, bs.overcast
	, CASE WHEN bs.wind = 'Indoors' THEN 0
        ELSE CAST(SUBSTRING_INDEX(bs.wind, ' ', 1) AS UNSIGNED)
    END AS wind
	, bs.winddir
	, CASE
		WHEN (bs.away_runs <= bs.home_runs AND bs.winner_home_or_away = 'A') OR (bs.away_runs <= bs.home_runs AND bs.winner_home_or_away = 'H') THEN 1
		WHEN (bs.away_runs >= bs.home_runs AND bs.winner_home_or_away = 'H') OR (bs.away_runs >= bs.home_runs AND bs.winner_home_or_away = 'A') THEN 0
	END AS winner_home
	, g.stadium_id AS stadium_id
FROM game g
    INNER JOIN boxscore bs
        ON g.game_id = bs.game_id
WHERE bs.winner_home_or_away != ''
;

-- Creating index for quick joins on im_game_table
CREATE INDEX idx_home_pitcher_game_date_im_game_table ON im_game_table(home_pitcher, game_date);
CREATE INDEX idx_away_pitcher_game_date_im_game_table ON im_game_table(away_pitcher, game_date);
CREATE INDEX idx_home_team_id_game_date_im_game_table ON im_game_table(home_team_id, game_date);
CREATE INDEX idx_away_team_id_game_date_im_game_table ON im_game_table(away_team_id, game_date);


-- Creating intermediate pitcher table
DROP TABLE IF EXISTS im_pitcher_table;

CREATE TABLE im_pitcher_table AS
SELECT
    pc.game_id AS game_id
    , g.local_date AS local_date
    , pc.pitcher AS pitcher
    , pc.Walk  AS Walk
    , pc.Strikeout AS Strikeout
    , pc.startingPitcher AS startingPitcher
    , pc.atBat AS atBat
    , pc.hit AS hit
    , pc.plateApperance AS plateApperance
    , pc.outsPlayed AS outsPlayed
    , pc.pitchesThrown AS pitchesThrown
    , pc.DaysSinceLastPitch AS DaysSinceLastPitch
    , pc.Home_Run AS Home_Run
    , pc.Hit_By_Pitch AS Hit_By_Pitch
FROM pitcher_counts pc
    INNER JOIN game g ON pc.game_id = g.game_id
ORDER BY local_date
;

-- Creating index for quick joins on im_pitcher_table
CREATE INDEX idx_pitcher_local_date_pitcher_table ON im_pitcher_table(pitcher, local_date);


-- Creating the intermediate team batting counts
DROP TABLE IF EXISTS im_team_batting_counts;

-- Creating intermediate team batting counts table
CREATE TABLE im_team_batting_counts AS
SELECT
    tbc.game_id AS game_id
    , tbc.team_id AS team_id
    , tbc.atbat AS atbat
    , tbc.hit AS hit
    , tbc.Hit_By_Pitch AS Hit_By_Pitch
    , tbc.Walk AS Walk
    , tbc.Sac_Fly AS Sac_Fly
    , tbc.Single AS Single
    , tbc.`Double` AS `Double`
    , tbc.Triple AS Triple
    , tbc.Home_Run AS Home_Run
    , tbc.Strikeout AS Strikeout
    , g.local_date AS local_date
FROM team_batting_counts tbc
    INNER JOIN game g ON tbc.game_id = g.game_id
ORDER BY local_date
;

-- Creating index for quick joins on im_team_batting_counts
CREATE INDEX idx_team_id_local_date_team_batting_counts ON im_team_batting_counts(team_id, local_date);


-- Creating Pitchers Feature Table

-- Feature 1: Creating rolling Average Strikeout to Walk Ratio for 100 days
-- https://en.wikipedia.org/wiki/Strikeout-to-walk_ratio
-- Strikeout to Walk Ratio = Number of Strikeouts/No of Walks

-- Feature 2: Creating rolling opponents batting average for 100 days againts the pitcher
-- Please note that the hits and atbats in this table are taken from pitcher's stats and not from batter's stats

-- Feature 3: Creating rolling pitcher strikeout rate for 100 days
-- https://library.fangraphs.com/offense/rate-stats/

-- Feature 4: Creating rolling outsplayed per pitches thrown ratio for 100 days
-- Self created feature: This feature means amongst the total pitches thrown, how many of them were converted into a out

-- Feature 5: Power Finesse Ratio (Strikeouts + Walks)/Innings Pitched
-- Innings Pitched = outsplayed/3
-- https://en.wikipedia.org/wiki/Power_finesse_ratio

-- Feature 6: Walks and Hits per innings pitched
-- https://en.wikipedia.org/wiki/Walks_plus_hits_per_inning_pitched

-- Feature 7: DICE Defense-Independent Component ERA
-- https://en.wikipedia.org/wiki/Defense-Independent_Component_ERA


DROP TABLE IF EXISTS fl_pitchers_feature_table;
-- Rolling Average Strikeout to Walk Ratio
CREATE TABLE fl_pitchers_feature_table AS
SELECT
    ipt1.game_id
    , DATE(ipt1.local_date) AS game_date
    , ipt1.pitcher
    , ipt1.Strikeout
    , ipt1.Walk
    , ipt1.hit
    , ipt1.atBat
    , ipt1.plateApperance
    , ipt1.outsPlayed
    , ipt1.outsPlayed / 3 AS innings
    , ipt1.Home_Run
    , ipt1.Hit_By_Pitch
    , COALESCE(SUM(ipt2.Strikeout) / NULLIF(SUM(ipt2.Walk), 0), 0) AS pitcher_strikeout_to_walk_ratio
    , COALESCE(SUM(ipt2.hit) / NULLIF(SUM(ipt2.atBat), 0), 0) AS pitcher_opponent_batting_average
    , COALESCE(SUM(ipt2.Strikeout) / NULLIF(SUM(ipt2.plateApperance), 0), 0) AS pitcher_strikeout_rate
    , COALESCE(SUM(ipt2.outsPlayed) / NULLIF(SUM(ipt2.pitchesThrown), 0), 0) AS pitcher_outsplayed_per_pitches_rate
    , COALESCE(SUM(ipt2.Strikeout + ipt2.Walk) / NULLIF(SUM(ipt2.outsPlayed / 3), 0), 0) AS pitcher_power_finesse_ratio
    , COALESCE(SUM(ipt2.Walk + ipt2.hit) / NULLIF(SUM(ipt2.outsPlayed / 3), 0), 0) AS pitcher_walks_hits_per_innings_pitched
    , COALESCE(SUM(13 * (ipt2.Home_Run) + 3 * (ipt2.Walk + ipt2.Hit_By_Pitch) - 2 * (ipt2.Strikeout)) / NULLIF(SUM(ipt2.outsPlayed / 3), 0), 0) AS pitcher_dice
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;


-- Creating index for quick joins on fl_pitchers_feature_table
CREATE INDEX idx_pitchers_game_date_fl_pitchers_feature_table ON fl_pitchers_feature_table(pitcher, game_date);



-- Team Overall Batting Features

-- Feature 8: Rolling batting average of the entire team before the match
-- https://en.wikipedia.org/wiki/Batting_average_(baseball)

-- Feature 9: Rolling On Base Percentage OBP of the entire team before the match
-- https://en.wikipedia.org/wiki/On-base_percentage


-- Feature 10: Rolling Slugging Percentage SLG of the entire team before the match
-- https://en.wikipedia.org/wiki/Slugging_percentage

-- Feature 11: Gross Production Average GPA
-- https://en.wikipedia.org/wiki/Gross_production_average

-- Feature 12: Team Walk to Strikeout ratio
-- https://en.wikipedia.org/wiki/Walk-to-strikeout_ratio


DROP TABLE IF EXISTS fl_team_batting_feature_table;
-- Rolling Batting Average for a Specific Batter
CREATE TABLE fl_team_batting_feature_table AS
SELECT
    itbc1.team_id
    , itbc1.game_id
    , DATE(itbc1.local_date) AS game_date
    , itbc1.hit
    , itbc1.atbat
    , itbc1.Walk
    , itbc1.Hit_By_Pitch
    , itbc1.Sac_Fly
    , itbc1.Single
    , itbc1.`Double`
    , itbc1.Triple
    , itbc1.Home_Run
    , COALESCE(SUM(itbc2.hit) / NULLIF(SUM(itbc2.atbat), 0), 0) AS team_bat_Avg
    , COALESCE(SUM(itbc2.hit + itbc2.Walk + itbc2.Hit_By_Pitch) / NULLIF(SUM(itbc2.atbat + itbc2.Walk + itbc2.Hit_By_Pitch + itbc2.Sac_Fly), 0), 0) AS team_on_base_percentage
    , COALESCE(SUM(itbc2.Single + 2 * itbc2.`Double`  + 3 * itbc2.Triple + 4 * itbc2.Home_Run) / NULLIF(SUM(itbc2.atbat), 0), 0) AS team_slugging_percentage
    , COALESCE(SUM(itbc2.Walk) / NULLIF(SUM(itbc2.Strikeout), 0), 0) AS team_walk_to_strikeout_ratio
FROM im_team_batting_counts itbc1
    LEFT JOIN im_team_batting_counts itbc2
        ON
            (((DATEDIFF(DATE(itbc1.local_date), DATE(itbc2.local_date)) <= 100) AND (DATEDIFF(DATE(itbc1.local_date)
                , DATE(itbc2.local_date)) > 0)) AND (itbc1.team_id = itbc2.team_id))

GROUP BY itbc1.team_id, DATE(itbc1.local_date)
ORDER BY itbc1.team_id, DATE(itbc1.local_date)
;



-- Creating index for quick joins on fl_team_batting_feature_table
CREATE INDEX idx_team_id_game_date_fl_team_batting_features ON fl_team_batting_feature_table(team_id, game_date);


-- Creating the final joined feature table
DROP TABLE IF EXISTS fl_features_table;
CREATE TABLE fl_features_table AS
SELECT
	igt.game_id AS game_id
	, igt.game_date AS game_date
	, igt.home_pitcher AS home_pitcher
	, igt.home_team_id AS home_team_id
	, igt.away_pitcher AS away_pitcher
	, igt.away_team_id AS away_team_id
	, igt.temperature AS temperature
	, igt.overcast AS overcast
	, igt.wind AS wind
	, igt.winddir AS winddir
	, igt.stadium_id AS stadium_id
	, fpft_home.pitcher_strikeout_to_walk_ratio AS home_pitcher_strikeout_to_walk_ratio
	, fpft_home.pitcher_opponent_batting_average AS home_pitcher_opponent_batting_average
	, fpft_home.pitcher_strikeout_rate AS home_pitcher_strikeout_rate
	, fpft_home.pitcher_outsplayed_per_pitches_rate AS home_pitcher_outsplayed_per_pitches_rate
	, fpft_home.pitcher_power_finesse_ratio AS home_pitcher_power_finesse_ratio
	, fpft_home.pitcher_walks_hits_per_innings_pitched AS home_pitcher_walks_hits_per_innings_pitched
	, fpft_home.pitcher_dice AS home_pitcher_dice
	, ftbft_home.team_bat_Avg AS home_team_bat_Avg
	, ftbft_home.team_on_base_percentage AS home_team_on_base_percentage
	, ftbft_home.team_slugging_percentage AS home_team_slugging_percentage
	, ftbft_home.team_walk_to_strikeout_ratio AS home_team_walk_to_strikeout_ratio
	, fpft_away.pitcher_strikeout_to_walk_ratio AS away_pitcher_strikeout_to_walk_ratio
	, fpft_away.pitcher_opponent_batting_average AS away_pitcher_opponent_batting_average
	, fpft_away.pitcher_strikeout_rate AS away_pitcher_strikeout_rate
	, fpft_away.pitcher_outsplayed_per_pitches_rate AS away_pitcher_outsplayed_per_pitches_rate
	, fpft_away.pitcher_power_finesse_ratio AS away_pitcher_power_finesse_ratio
	, fpft_away.pitcher_walks_hits_per_innings_pitched AS away_pitcher_walks_hits_per_innings_pitched
	, fpft_away.pitcher_dice AS away_pitcher_dice
	, ftbft_away.team_bat_Avg AS away_team_bat_Avg
	, ftbft_away.team_on_base_percentage AS away_team_on_base_percentage
	, ftbft_away.team_slugging_percentage AS away_team_slugging_percentage
	, ftbft_away.team_walk_to_strikeout_ratio AS away_team_walk_to_strikeout_ratio
	, igt.winner_home AS winner_home
FROM im_game_table igt
    LEFT JOIN fl_pitchers_feature_table fpft_home
        ON ((igt.home_pitcher = fpft_home.pitcher) AND (igt.game_date = fpft_home.game_date))
    LEFT JOIN fl_team_batting_feature_table ftbft_home
        ON ((igt.home_team_id = ftbft_home.team_id) AND (igt.game_date = ftbft_home.game_date))
	LEFT JOIN fl_pitchers_feature_table fpft_away
        ON ((igt.away_pitcher = fpft_away.pitcher) AND (igt.game_date = fpft_away.game_date))
	LEFT JOIN fl_team_batting_feature_table ftbft_away
        ON ((igt.away_team_id = ftbft_away.team_id) AND (igt.game_date = ftbft_away.game_date))
;
