-- Should compare these features between home team and away team

-- Combining game and box_score table
CREATE TABLE im_game_table AS
SELECT g.game_id, g.local_date, g.home_pitcher, g.home_team_id, g.away_pitcher, g.away_team_id, bs.winner_home_or_away
FROM game g
    INNER JOIN boxscore bs
        ON g.game_id = bs.game_id
;

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


-- Feature 1: Creating rolling Average Strikeout to Walk Ratio for 100 days
-- https://en.wikipedia.org/wiki/Strikeout-to-walk_ratio
-- Strikeout to Walk Ratio = Number of Strikeouts/No of Walks
DROP TABLE IF EXISTS fl_rolling_strikeout_to_walk_ratio;
-- Rolling Average Strikeout to Walk Ratio
CREATE TABLE fl_rolling_strikeout_to_walk_ratio AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.Strikeout
    , ipt1.Walk
    , COALESCE(SUM(ipt2.Strikeout) / NULLIF(SUM(ipt2.Walk), 0), 0) AS strikeout_to_walk_ratio
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;


-- Feature 2: Creating rolling opponents batting average for 100 days againts the pitcher
-- Please note that the hits and atbats in this table are taken from pitcher's stats and not from batter's stats
DROP TABLE IF EXISTS fl_rolling_opponent_batting_average;
-- Rolling Opponent Batting Average
CREATE TABLE fl_rolling_opponent_batting_average AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.hit
    , ipt1.atBat
    , COALESCE(SUM(ipt2.hit) / NULLIF(SUM(ipt2.atBat), 0), 0) AS opponent_rolling_batting_average
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date), DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;

-- Feature 3: Creating rolling pitcher strikeout rate for 100 days
-- https://library.fangraphs.com/offense/rate-stats/
DROP TABLE IF EXISTS fl_rolling_pitcher_strikeout_rate;

-- Rolling Opponent Batting Average
CREATE TABLE fl_rolling_pitcher_strikeout_rate AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.Strikeout
    , ipt1.plateApperance
    , COALESCE(SUM(ipt2.Strikeout) / NULLIF(SUM(ipt2.plateApperance), 0), 0) AS rolling_strikeout_rate
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date), DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;


-- Feature 4: Creating rolling outsplayed per pitches thrown ratio for 100 days
-- Self created feature: This feature means amongst the total pitches thrown, how many of them were converted into a out
DROP TABLE IF EXISTS fl_rolling_outsplayed_per_pitches_thrown_ratio;
-- Rolling Opponent Batting Average
CREATE TABLE fl_rolling_outsplayed_per_pitches_thrown_ratio AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.outsPlayed
    , ipt1.pitchesThrown
    , COALESCE(SUM(ipt2.outsPlayed) / NULLIF(SUM(ipt2.pitchesThrown), 0), 0) AS rolling_outsplayed_per_pitches_rate
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date), DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;


-- Feature 5: Power Finesse Ratio (Strikeouts + Walks)/Innings Pitched
-- Innings Pitched = outsplayed/3
-- https://en.wikipedia.org/wiki/Power_finesse_ratio
DROP TABLE IF EXISTS fl_rolling_power_finesse_ratio;
-- Rolling Power Finesse Ratio
CREATE TABLE fl_rolling_power_finesse_ratio AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.Strikeout
    , ipt1.Walk
    , ipt1.outsPlayed
    , COALESCE(SUM(ipt2.Strikeout + ipt2.Walk) / NULLIF(SUM(ipt2.outsPlayed / 3), 0), 0) AS rolling_power_finesse_ratio
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date), DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;

-- Feature 6: Walks and Hits per innings pitched
-- https://en.wikipedia.org/wiki/Walks_plus_hits_per_inning_pitched
DROP TABLE IF EXISTS fl_rolling_walks_hits_per_innings_pitched_ratio;
CREATE TABLE fl_rolling_walks_hits_per_innings_pitched_ratio AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.Walk
    , ipt1.hit
    , ipt1.outsPlayed / 3 AS innings
    , COALESCE(SUM(ipt2.Walk + ipt2.hit) / NULLIF(SUM(ipt2.outsPlayed / 3), 0), 0) AS rolling_walks_hits_per_innings_pitched
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date), DATE(ipt2.local_date)) <= 100) AND (DATEDIFF(DATE(ipt1.local_date)
                , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;

-- Feature 7: Rolling batting average of the entire team before the match
-- https://en.wikipedia.org/wiki/Batting_average_(baseball)
DROP TABLE IF EXISTS fl_rolling_batting_average;
-- Rolling Batting Average for a Specific Batter
CREATE TABLE fl_rolling_batting_average AS
SELECT
    ibba1.batter
    , ibba1.team_id
    , ibba1.hit
    , ibba1.atbat
    , COALESCE(SUM(ibba2.hit) / NULLIF(SUM(ibba2.atbat), 0), 0) AS bat_Avg
    , DATE(ibba1.local_date) AS game_date
FROM im_batter_batting_average ibba1
    LEFT JOIN im_batter_batting_average ibba2
        ON
            (((DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) <= 100) AND (DATEDIFF(DATE(ibba1.local_date)
                , DATE(ibba2.local_date)) > 0)) AND (ibba1.batter = ibba2.batter))

GROUP BY ibba1.batter, DATE(ibba1.local_date)
ORDER BY ibba1.batter, DATE(ibba1.local_date)
;


DROP TABLE IF EXISTS fl_team_rolling_batting_average;
-- Rolling Batting Average for a the entire team
CREATE TABLE fl_team_rolling_batting_average AS
SELECT
    team_id
    , game_date
    , AVG(bat_avg) AS team_rolling_bat_Avg
FROM fl_rolling_batting_average
GROUP BY team_id, game_date
;

-- Feature 8: DICE Defense-Independent Component ERA
-- https://en.wikipedia.org/wiki/Defense-Independent_Component_ERA
DROP TABLE IF EXISTS fl_rolling_pitcher_dice;
CREATE TABLE fl_rolling_pitcher_dice AS
SELECT
    ipt1.game_id
    , ipt1.pitcher
    , ipt1.Walk
    , ipt1.Home_Run
    , ipt1.Hit_By_Pitch
    , ipt1.hit
    , ipt1.outsPlayed / 3 AS innings
    , ipt1.Strikeout
    , COALESCE(SUM(13 * (ipt2.Home_Run) + 3 * (ipt2.Walk + ipt2.Hit_By_Pitch) - 2 * (ipt2.Strikeout)) / NULLIF(SUM(ipt2.outsPlayed / 3), 0), 0) AS rolling_pitcher_dice
    , DATE(ipt1.local_date) AS game_date
FROM im_pitcher_table ipt1
    LEFT JOIN im_pitcher_table ipt2
        ON
            (((DATEDIFF(DATE(ipt1.local_date), DATE(ipt2.local_date)) <= 100)
                AND (DATEDIFF(DATE(ipt1.local_date)
                    , DATE(ipt2.local_date)) > 0)) AND (ipt1.pitcher = ipt2.pitcher))
GROUP BY ipt1.pitcher, DATE(ipt1.local_date)
ORDER BY ipt1.pitcher, DATE(ipt1.local_date)
;

-- Feature 9: Rolling On Base Percentage OBP of the entire team before the match
-- https://en.wikipedia.org/wiki/On-base_percentage
DROP TABLE IF EXISTS fl_rolling_on_base_percentage;

CREATE TABLE fl_rolling_on_base_percentage AS
SELECT
    ibba1.batter
    , ibba1.team_id
    , ibba1.hit
    , ibba1.Walk
    , ibba1.atbat
    , ibba1.Hit_By_Pitch
    , ibba1.Sac_Fly
    , COALESCE(SUM(ibba2.hit + ibba2.Walk + ibba2.Hit_By_Pitch) / NULLIF(SUM(ibba2.atbat + ibba2.Walk + ibba2.Hit_By_Pitch + ibba2.Sac_Fly), 0), 0) AS on_base_percentage
    , DATE(ibba1.local_date) AS game_date
FROM im_batter_batting_average ibba1
    LEFT JOIN im_batter_batting_average ibba2
        ON
            (((DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) <= 100) AND (DATEDIFF(DATE(ibba1.local_date)
                , DATE(ibba2.local_date)) > 0)) AND (ibba1.batter = ibba2.batter))

GROUP BY ibba1.batter, DATE(ibba1.local_date)
ORDER BY ibba1.batter, DATE(ibba1.local_date)
;


DROP TABLE IF EXISTS fl_team_rolling_on_base_percentage;
-- Rolling on base percentage for the entire team
CREATE TABLE fl_team_rolling_on_base_percentage AS
SELECT
    team_id
    , game_date
    , AVG(on_base_percentage) AS team_rolling_on_base_percentage
FROM fl_rolling_on_base_percentage
GROUP BY team_id, game_date
;


-- Feature 10: Rolling Slugging Percentage SLG of the entire team before the match
-- https://en.wikipedia.org/wiki/Slugging_percentage
DROP TABLE IF EXISTS fl_rolling_slugging_percentage;

CREATE TABLE fl_rolling_slugging_percentage AS
SELECT
    ibba1.batter
    , ibba1.team_id
    , ibba1.Single
    , ibba1.`Double`
    , ibba1.Triple
    , ibba1.Home_Run
    , ibba1.atbat
    , COALESCE(SUM(ibba2.Single + 2 * ibba2.`Double`  + 3 * ibba2.Triple + 4 * ibba2.Home_Run) / NULLIF(SUM(ibba2.atbat), 0), 0) AS slugging_percentage
    , DATE(ibba1.local_date) AS game_date
FROM im_batter_batting_average ibba1
    LEFT JOIN im_batter_batting_average ibba2
        ON
            (((DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) <= 100) AND (DATEDIFF(DATE(ibba1.local_date)
                , DATE(ibba2.local_date)) > 0)) AND (ibba1.batter = ibba2.batter))

GROUP BY ibba1.batter, DATE(ibba1.local_date)
ORDER BY ibba1.batter, DATE(ibba1.local_date)
;

DROP TABLE IF EXISTS fl_team_rolling_slugging_percentage;
-- Rolling Slugging Percentage for the entire team
CREATE TABLE fl_team_rolling_slugging_percentage AS
SELECT team_id, game_date, AVG(slugging_percentage) AS team_rolling_slugging_percentage
FROM fl_rolling_slugging_percentage
GROUP BY team_id, game_date
;

-- Creating index for quick joins on im_game_table
CREATE INDEX idx_imgame_gameid ON im_game_table(game_id);
CREATE INDEX idx_imgame_pitcher ON im_game_table(home_pitcher);
CREATE INDEX idx_imgame_date ON im_game_table(local_date);


-- Getting Sample Results for 100 days rolling strikeout to walk ratio for home team pitcher
DROP TABLE IF EXISTS fl_pitcher_feature_table;
CREATE TABLE fl_pitcher_feature_table AS
SELECT
    igt.game_id
    , igt.local_date
    , igt.home_pitcher
    , igt.home_team_id
    , frstwr.strikeout_to_walk_ratio
    , igt.winner_home_or_away
FROM im_game_table igt
    LEFT JOIN fl_rolling_strikeout_to_walk_ratio frstwr
        ON ((igt.game_id = frstwr.game_id) AND ((DATE(igt.local_date) = DATE(frstwr.game_date))
            AND (igt.home_pitcher = frstwr.pitcher)))
;
