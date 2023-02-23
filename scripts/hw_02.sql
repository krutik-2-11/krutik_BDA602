DROP TABLE IF EXISTS batter_batting_average;

CREATE TABLE batter_batting_average AS
SELECT bc.game_id AS game_id, bc.batter AS batter, bc.atbat AS atbat, bc.hit AS hit, g.local_date AS local_date, g.et_date AS et_date
FROM batter_counts bc
    INNER JOIN game g ON bc.game_id = g.game_id
ORDER BY local_date
;

-- Historical Batting Average for batters
SELECT bba.batter, SUM(bba.hit) / SUM(bba.atbat) AS bat_Avg
FROM batter_batting_average bba
GROUP BY bba.batter
;
-- Annual Batting Average for batters
SELECT bba.batter, SUM(bba.hit) / SUM(bba.atbat) AS bat_Avg, YEAR(bba.local_date) AS game_year
FROM batter_batting_average bba
GROUP BY bba.batter, YEAR(bba.local_date)
;
-- Replacing Null values with Zeros and addition of Index WIP
-- Rolling Batting Average for a Specific Batter
SELECT bba1.batter, bba1.hit, bba1.atbat, SUM(bba2.hit) / SUM(bba2.atbat) AS batting_average, DATE(bba1.local_date) AS local_date1, DATE(bba2.local_date) AS local_date2, DATEDIFF(DATE(bba1.local_date), DATE(bba2.local_date)) AS date_diff
FROM batter_batting_average bba1
    LEFT JOIN batter_batting_average bba2
        ON
            (DATEDIFF(DATE(bba1.local_date), DATE(bba2.local_date)) <= 100) AND (DATEDIFF(DATE(bba1.local_date), DATE(bba2.local_date)) > 0) AND (bba1.batter = bba2.batter)
WHERE bba1.batter = 407832
GROUP BY DATE(bba1.local_date)
ORDER BY DATE(bba1.local_date), date_diff DESC
;
