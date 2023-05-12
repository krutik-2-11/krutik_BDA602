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

# Baseball Features
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

