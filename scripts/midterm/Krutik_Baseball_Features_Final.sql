USE baseball;

-- Should compare these features between home team and away team

-- Combining game and box_score table
DROP TABLE IF EXISTS im_game_table;
CREATE TABLE im_game_table AS
SELECT
	g.game_id
	, g.local_date
	, g.home_pitcher
	, g.home_team_id
	, g.away_pitcher
	, g.away_team_id
	, CASE WHEN  bs.winner_home_or_away = 'H' THEN 1 ELSE 0 END AS winner_home
FROM game g
    INNER JOIN boxscore bs
        ON g.game_id = bs.game_id
;

-- Creating index for quick joins on im_game_table
CREATE INDEX idx_imgame_gameid ON im_game_table(game_id);
CREATE INDEX idx_imgame_pitcher ON im_game_table(home_pitcher);
CREATE INDEX idx_imgame_date ON im_game_table(local_date);


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
CREATE INDEX idx_pitcher ON im_pitcher_table(pitcher);
CREATE INDEX idx_date ON im_pitcher_table(local_date);


-- Creating the intermediate batter table
DROP TABLE IF EXISTS im_batter_batting_average;

-- Creating intermediate batting table
CREATE TABLE im_batter_batting_average AS
SELECT
    bc.game_id AS game_id
    , bc.team_id AS team_id
    , bc.batter AS batter
    , bc.atbat AS atbat
    , bc.hit AS hit
    , bc.Hit_By_Pitch AS Hit_By_Pitch
    , bc.Walk AS Walk
    , bc.Sac_Fly AS Sac_Fly
    , bc.Single AS Single
    , bc.`Double` AS `Double`
    , bc.Triple AS Triple
    , bc.Home_Run AS Home_Run
    , g.local_date AS local_date
    , g.et_date AS et_date
FROM batter_counts bc
    INNER JOIN game g ON bc.game_id = g.game_id
ORDER BY local_date
;

-- Creating index for quick joins on im_batter_batting_average
CREATE INDEX idx_batter ON im_batter_batting_average(batter);
CREATE INDEX idx_date ON im_batter_batting_average(local_date);


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
CREATE INDEX idx_pitchers_pitcher ON fl_pitchers_feature_table(pitcher);
CREATE INDEX idx_pitchers_game_date ON fl_pitchers_feature_table(game_date);


-- Team Overall Batting Features

-- Feature 8: Rolling batting average of the entire team before the match
-- https://en.wikipedia.org/wiki/Batting_average_(baseball)

-- Feature 9: Rolling On Base Percentage OBP of the entire team before the match
-- https://en.wikipedia.org/wiki/On-base_percentage


-- Feature 10: Rolling Slugging Percentage SLG of the entire team before the match
-- https://en.wikipedia.org/wiki/Slugging_percentage

-- First getting for individual batters
DROP TABLE IF EXISTS fl_batter_feature_table;
-- Rolling Batting Average for a Specific Batter
CREATE TABLE fl_batter_feature_table AS
SELECT
    ibba1.batter
    , ibba1.game_id
    , ibba1.team_id
    , DATE(ibba1.local_date) AS game_date
    , ibba1.hit
    , ibba1.atbat
    , ibba1.Walk
    , ibba1.Hit_By_Pitch
    , ibba1.Sac_Fly
    , ibba1.Single
    , ibba1.`Double`
    , ibba1.Triple
    , ibba1.Home_Run
    , COALESCE(SUM(ibba2.hit) / NULLIF(SUM(ibba2.atbat), 0), 0) AS bat_Avg
    , COALESCE(SUM(ibba2.hit + ibba2.Walk + ibba2.Hit_By_Pitch) / NULLIF(SUM(ibba2.atbat + ibba2.Walk + ibba2.Hit_By_Pitch + ibba2.Sac_Fly), 0), 0) AS on_base_percentage
    , COALESCE(SUM(ibba2.Single + 2 * ibba2.`Double`  + 3 * ibba2.Triple + 4 * ibba2.Home_Run) / NULLIF(SUM(ibba2.atbat), 0), 0) AS slugging_percentage
FROM im_batter_batting_average ibba1
    LEFT JOIN im_batter_batting_average ibba2
        ON
            (((DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) <= 100) AND (DATEDIFF(DATE(ibba1.local_date)
                , DATE(ibba2.local_date)) > 0)) AND (ibba1.batter = ibba2.batter))

GROUP BY ibba1.batter, DATE(ibba1.local_date)
ORDER BY ibba1.batter, DATE(ibba1.local_date)
;


-- Getting the average stats for the entire team
DROP TABLE IF EXISTS fl_team_batting_features;
-- Rolling Batting Average for a the entire team
CREATE TABLE fl_team_batting_features AS
SELECT
    team_id
    , game_date
    , AVG(bat_avg) AS team_bat_Avg
    , AVG(on_base_percentage) AS team_on_base_percentage
    , AVG(slugging_percentage) AS team_slugging_percentage
FROM fl_batter_feature_table
GROUP BY team_id, game_date
;


-- Creating index for quick joins on fl_team_batting_features
CREATE INDEX idx_team_batting_team_id ON fl_team_batting_features(team_id);
CREATE INDEX idx_team_batting_game_date ON fl_team_batting_features(game_date);


-- Creating the final feature table
DROP TABLE IF EXISTS fl_features_table;
CREATE TABLE fl_features_table AS
SELECT
	igt.game_id AS game_id
	, igt.local_date AS game_date
	, igt.home_pitcher AS home_pitcher
	, igt.home_team_id AS home_team_id
	, igt.away_pitcher AS away_pitcher
	, igt.away_team_id AS away_team_id
	, fpft_home.pitcher_strikeout_to_walk_ratio AS home_pitcher_strikeout_to_walk_ratio
	, fpft_home.pitcher_opponent_batting_average AS home_pitcher_opponent_batting_average
	, fpft_home.pitcher_strikeout_rate AS home_pitcher_strikeout_rate
	, fpft_home.pitcher_outsplayed_per_pitches_rate AS home_pitcher_outsplayed_per_pitches_rate
	, fpft_home.pitcher_power_finesse_ratio AS home_pitcher_power_finesse_ratio
	, fpft_home.pitcher_walks_hits_per_innings_pitched AS home_pitcher_walks_hits_per_innings_pitched
	, fpft_home.pitcher_dice AS home_pitcher_dice
	, ftbf_home.team_bat_Avg AS home_team_bat_Avg
	, ftbf_home.team_on_base_percentage AS home_team_on_base_percentage
	, ftbf_home.team_slugging_percentage AS home_team_slugging_percentage
	, fpft_away.pitcher_strikeout_to_walk_ratio AS away_pitcher_strikeout_to_walk_ratio
	, fpft_away.pitcher_opponent_batting_average AS away_pitcher_opponent_batting_average
	, fpft_away.pitcher_strikeout_rate AS away_pitcher_strikeout_rate
	, fpft_away.pitcher_outsplayed_per_pitches_rate AS away_pitcher_outsplayed_per_pitches_rate
	, fpft_away.pitcher_power_finesse_ratio AS away_pitcher_power_finesse_ratio
	, fpft_away.pitcher_walks_hits_per_innings_pitched AS away_pitcher_walks_hits_per_innings_pitched
	, fpft_away.pitcher_dice AS away_pitcher_dice
	, ftbf_away.team_bat_Avg AS away_team_bat_Avg
	, ftbf_away.team_on_base_percentage AS away_team_on_base_percentage
	, ftbf_away.team_slugging_percentage AS away_team_slugging_percentage
	, igt.winner_home AS winner_home
FROM im_game_table igt
    LEFT JOIN fl_pitchers_feature_table fpft_home
        ON ((igt.home_pitcher = fpft_home.pitcher) AND (DATE(igt.local_date) = fpft_home.game_date))
	LEFT JOIN fl_team_batting_features ftbf_home
        ON ((igt.home_team_id = ftbf_home.team_id) AND (DATE(igt.local_date) = ftbf_home.game_date))
    LEFT JOIN fl_pitchers_feature_table fpft_away
        ON ((igt.away_pitcher = fpft_away.pitcher) AND (DATE(igt.local_date) = fpft_away.game_date))
    LEFT JOIN fl_team_batting_features ftbf_away
        ON ((igt.away_team_id = ftbf_away.team_id) AND (DATE(igt.local_date) = ftbf_away.game_date))
;
