# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Run Project
- Clone the repository using `git clone https://github.com/krutik-2-11/krutik_BDA602.git`
- Run `docker-compose build` 
- Run `docker-compose up`
- After docker container is finished, it will create an `krutik_pathak_midterm_file.html` that contains the complete feature engineering report and model evaluation. 

# Introduction
- The purpose of the project is to predict if the home team will win a baseball game considering different factors such as starting pitcher statistics, team batting statistics and other external   conditions.
- Different features have been buiilt based on pitcher's, batters statistics and other game conditions.
- The features are then evaluated based on statistical measures like mean of response plots, response-predictor violin plots, correlation heatmaps, and brute force analysis.
- Once the feature engineering is completed, different models such as logistic regression, decision trees, KNN, and support vector machines are trained and evaluated on different combination of   features.
- The performance of the models is evaluated using metrics such as accuracy, ROC curve, precision, recall.
- The model with best performance is then used for prediction of future games.

# Baseball Dataset
[Baseball Dataset](https://teaching.mrsharky.com/data/baseball.sql.tar.gz)
- The baseball dataset is loaded through a `baseball.sql` file, it is a 1.2 GB file.
- The dataset has tables for individual batting, pitching statistics and also for the team batting, pitching statistics, game summary, fielding statistics.
- Dataset Issues
  * The dataset does not have a metadata, so interpretation of several columns is difficult for someone who is not used to baseball game.
  * Some columns had only value `0` like `caughtStealing2B`, `caughtStealing3B`, `caughtStealingHome` in `pitchers_counts` table.
  * Some tables had the multiple instances of same value, for ex: in `force_out` and `Forceout` in `pitchers_counts` table.  

# 1. Baseball Features
- The source of truth is a mariadb database `baseball`. The database is loaded with a [SQL script](https://teaching.mrsharky.com/data/baseball.sql.tar.gz)
- Once the database is loaded with `baseball.sql`, I built my features using the script [Krutik_Baseball_Features_Final.sql](https://github.com/krutik-2-11/krutik_BDA602/blob/final/scripts/final/Krutik_Baseball_Features_Final.sql)
- Some essential features that I built are:
  * [Strikeout to Walk Ratio](https://en.wikipedia.org/wiki/Strikeout-to-walk_ratio) : `K/BB`
  * [Opponents Batting Average] : `SUM(HIT)/SUM (AT BATS)`
  * [Pitcher Strikeout Rate](https://library.fangraphs.com/offense/rate-stats/) : `K/PA`
  * [Outsplayed per pitches thrown] : `SUM(Outsplayed)/SUM(Pitches Thrown)`
  * [Power Finesse Ratio](https://en.wikipedia.org/wiki/Power_finesse_ratio) : `SUM(Strikeouts + Walks)/Innings Pitched`
  * [WHIP](https://en.wikipedia.org/wiki/Walks_plus_hits_per_inning_pitched) : `SUM(Walks + Hits)/Innings Pitched`
  * [DICE](https://en.wikipedia.org/wiki/Defense-Independent_Component_ERA) : `(13*Home Runs + 3(Walks + Hit Batters) -2*Strikeouts)/Innings Pitched`
  * [Team Batting Average](https://en.wikipedia.org/wiki/Batting_average_(baseball)) : `SUM(Hits)/SUM(At Bats)`
  * [Team On base Percentage](https://en.wikipedia.org/wiki/On-base_percentage) : `SUM(Hits + Walks + Hit By Pitch)/SUM(At Bat + Walks + Hit by Pitch + Sacrifice Fly)`
  * [Team Slugging Percentage](https://en.wikipedia.org/wiki/Slugging_percentage) : `(1*(1B) + 2*(2B) + 3*(3B) + 4*(Home Runs))/At Bats`
  * [Team Gross Production Average](https://en.wikipedia.org/wiki/Gross_production_average) : `(1.8*On Base Plus Slugging + Slugging Percentage)/4`
  * [Team Walk to Strikout Ratio](https://en.wikipedia.org/wiki/Walk-to-strikeout_ratio) : `BB/K`

- There are other features that I have taken into consideration from boxscore table : 
  * `Winddir` : Wind Direction
  * `Temperature` : temperature
  * `Wind Speed` : Wind Speed
  * `Overcast` : Overcast

# Rolling 100 days average
- I have taken rolling 100 days average as it gives a better estimate of recent performances of the players. Historic averages many times do not give the right estimate of the performance of     the player. 
- The place where rolling 100 days average does not work very well is when a player has an amazing stats in very recent times. For ex: in last 10 days, the player has been outstading but over   the 100 days span the stats are not very enticing. 

# Comparing the Home and Away sides
- I have taken the difference between the home feature and away feature to get the final feature. For ex: **Strikeout_to_Walk_Ratio_difference** is a feature that is calculated by `Strikeout_to_Walk_Ratio_Home - Strikeout_to_Walk_Ratio_Away`

# 2. Feature Engineering
## 2.1 Response vs Predictor Statistics
- In this part of feature engineering, we built plots between response variable (dependent variable) and predictors (independent variables) to indentify relationship between them. I used p-     value and student t-test (t-value) estimates to identify the underlying relationship. The violin plots and scatter plots are also built to get the intuition of the relationship. Additionally   mean of response, both weighted and unweighted, and random forest variable importance is used for further analysis.
  
  * [p-value](https://en.wikipedia.org/wiki/P-value): The p-value tells you how likely it is to observe the data by chance alone. A small p-value (typically less than 0.05) suggests strong evidence against the null hypothesis, indicating that the observed effect is unlikely to be due to random chance.
  * [t-value](https://www.geeksforgeeks.org/t-test/): A t-test is a type of inferential statistic used to determine if there is a significant difference between the means of two groups, which may be related in certain features.
  * Generally a small p-value (less than 0.05) and large magnitude of t-value (>=2.00 and <= -2.00) is [considered good for confidence](https://www.allbusiness.com/barrons_dictionary/dictionary-t-value-4942040-1.html#:~:text=Generally%2C%20any%20t%2Dvalue%20greater,the%20coefficient%20as%20a%20predictor.)
  * [Difference with Mean of Response](https://en.wikipedia.org/wiki/Variance_of_the_mean_and_predicted_responses): The difference with mean of response plots are used to find patterns in data. [For ex: If the age in the data increases, how the survival rate changes](https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/10/0/7). It is often analyzed using statistical tests to determine if the observed difference is statistically significant or if it could have occurred by chance. 
  ** Sometimes the population distribution is not uniform in different samples, in that case we take weighted mean of response. [Weighted mean of response gives better estimate of non uniformly spread data.](https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/4)

