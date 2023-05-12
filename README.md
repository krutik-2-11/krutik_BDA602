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
