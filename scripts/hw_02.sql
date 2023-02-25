DROP TABLE IF EXISTS im_batter_batting_average;

-- Creating intermediate batting table
CREATE TABLE im_batter_batting_average AS
SELECT bc.game_id AS game_id, bc.batter AS batter, bc.atbat AS atbat, bc.hit AS hit, g.local_date AS local_date, g.et_date AS et_date
FROM batter_counts bc
    INNER JOIN game g ON bc.game_id = g.game_id
ORDER BY local_date
;

-- Creating index for quick joins
CREATE INDEX idx_batter ON im_batter_batting_average(batter);
CREATE INDEX idx_date ON im_batter_batting_average(local_date);


DROP TABLE IF EXISTS fl_historical_batting_average;
-- Final Historical Batting Average for batters
CREATE TABLE fl_historical_batting_average AS
SELECT ibba.batter, COALESCE(SUM(ibba.hit) / NULLIF(SUM(ibba.atbat), 0), 0) AS bat_Avg
FROM im_batter_batting_average ibba
GROUP BY ibba.batter
;


DROP TABLE IF EXISTS fl_annual_batting_average;
-- Annual Batting Average for batters
CREATE TABLE fl_annual_batting_average AS
SELECT ibba.batter, COALESCE(SUM(ibba.hit) / NULLIF(SUM(ibba.atbat), 0), 0) AS bat_Avg, YEAR(ibba.local_date) AS game_year
FROM im_batter_batting_average ibba
GROUP BY ibba.batter, YEAR(ibba.local_date)
;


DROP TABLE IF EXISTS fl_rolling_batting_average;
-- Rolling Batting Average for a Specific Batter
CREATE TABLE fl_rolling_batting_average AS
SELECT ibba1.batter, ibba1.hit, ibba1.atbat, COALESCE(SUM(ibba2.hit) / NULLIF(SUM(ibba2.atbat), 0), 0) AS bat_Avg, DATE(ibba1.local_date) AS game_date
FROM im_batter_batting_average ibba1
    LEFT JOIN im_batter_batting_average ibba2
        ON
            (((DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) <= 100) AND (DATEDIFF(DATE(ibba1.local_date), DATE(ibba2.local_date)) > 0)) AND (ibba1.batter = ibba2.batter))
GROUP BY ibba1.batter, DATE(ibba1.local_date)
ORDER BY ibba1.batter, DATE(ibba1.local_date)
;
