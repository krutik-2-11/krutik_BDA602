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
- Go to the directory `scripts/final`
- Add the `baseball.sql` file in the directory `scripts/final`
- Run `docker-compose build`
- Run `docker-compose up`
- After docker container is finished, it will create an `krutik_pathak_final_file.html` that contains the complete feature engineering report and model evaluation.

# Introduction

- The purpose of the project is to predict if the home team will win a baseball game considering different factors such as starting pitcher statistics, team batting statistics and other external conditions.
- Different features have been built based on pitcher's, batters statistics and other game conditions.
- The features are then evaluated based on statistical measures like mean of response plots, response-predictor violin plots, correlation heatmaps, and brute force analysis.
- Once the feature engineering is completed, different models such as logistic regression, decision trees, KNN, and support vector machines are trained and evaluated on different combination of features.
- The performance of the models is evaluated using metrics such as accuracy, ROC curve, precision, recall.
- The model with the best performance is then used for prediction of future games.

# Baseball Dataset

[Baseball Dataset](https://teaching.mrsharky.com/data/baseball.sql.tar.gz)

- The baseball dataset is loaded through a `baseball.sql` file, it is a 1.2 GB file.
- The dataset has tables for individual batting, pitching statistics and also for the team batting, pitching statistics, game summary, fielding statistics.
- Dataset Issues
  - The dataset does not have a metadata, so interpretation of several columns is difficult for someone who is not used to baseball game.
  - Some columns had only value `0` like `caughtStealing2B`, `caughtStealing3B`, `caughtStealingHome` in `pitchers_counts` table.
  - Some tables had the multiple instances of same value, for ex: in `force_out` and `Forceout` in `pitchers_counts` table.
  - The `boxscore` table has some discripancy in the final result, i.e., even though home team has scored more runs than away team still the away team is the winner and vice versa. I handled this issue in my SQL script.

# 1. Baseball Features

- The source of truth is a mariadb database `baseball`. The database is loaded with a [SQL script](https://teaching.mrsharky.com/data/baseball.sql.tar.gz)
- Once the database is loaded with `baseball.sql`, I built my features using the script [Krutik_Baseball_Features_Final.sql](https://github.com/krutik-2-11/krutik_BDA602/blob/final/scripts/final/Krutik_Baseball_Features_Final.sql)
- Some essential features that I built are:

  - [Strikeout to Walk Ratio](https://en.wikipedia.org/wiki/Strikeout-to-walk_ratio) : `K/BB`
  - [Opponents Batting Average] : `SUM(HIT)/SUM (AT BATS)`
  - [Pitcher Strikeout Rate](https://library.fangraphs.com/offense/rate-stats/) : `K/PA`
  - [Outsplayed per pitches thrown] : `SUM(Outsplayed)/SUM(Pitches Thrown)`
  - [Power Finesse Ratio](https://en.wikipedia.org/wiki/Power_finesse_ratio) : `SUM(Strikeouts + Walks)/Innings Pitched`
  - [WHIP](https://en.wikipedia.org/wiki/Walks_plus_hits_per_inning_pitched) : `SUM(Walks + Hits)/Innings Pitched`
  - [DICE](https://en.wikipedia.org/wiki/Defense-Independent_Component_ERA) : `(13*Home Runs + 3(Walks + Hit Batters) -2*Strikeouts)/Innings Pitched`
  - [Team Batting Average](<https://en.wikipedia.org/wiki/Batting_average_(baseball)>) : `SUM(Hits)/SUM(At Bats)`
  - [Team On base Percentage](https://en.wikipedia.org/wiki/On-base_percentage) : `SUM(Hits + Walks + Hit By Pitch)/SUM(At Bat + Walks + Hit by Pitch + Sacrifice Fly)`
  - [Team Slugging Percentage](https://en.wikipedia.org/wiki/Slugging_percentage) : `(1*(1B) + 2*(2B) + 3*(3B) + 4*(Home Runs))/At Bats`
  - [Team Gross Production Average](https://en.wikipedia.org/wiki/Gross_production_average) : `(1.8*On Base Plus Slugging + Slugging Percentage)/4`
  - [Team Walk to Strikout Ratio](https://en.wikipedia.org/wiki/Walk-to-strikeout_ratio) : `BB/K`

- There are other features that I have taken into consideration from boxscore table :
  - `Winddir` : Wind Direction
  - `Temperature` : temperature
  - `Wind Speed` : Wind Speed
  - `Overcast` : Overcast

# Rolling 100 days average

- I have taken rolling 100 days average as it gives a better estimate of recent performances of the players. Historic averages many times do not give the right estimate of the performance of the player.
- The place where rolling 100 days average does not work very well is when a player has an amazing stats in very recent times. For ex: in last 10 days, the player has been outstading but over the 100 days span the stats are not very enticing.

# Comparing the Home and Away sides

- I have taken the difference between the home feature and away feature to get the final feature. For ex: **Strikeout_to_Walk_Ratio_difference** is a feature that is calculated by `Strikeout_to_Walk_Ratio_Home - Strikeout_to_Walk_Ratio_Away`

# 2. Feature Engineering

## 2.1 Response vs Predictor Statistics

- In this part of feature engineering, we built plots between response variable (dependent variable) and predictors (independent variables) to indentify relationship between them. I used p- value and student t-test (t-value) estimates to identify the underlying relationship. The violin plots and scatter plots are also built to get the intuition of the relationship. Additionally mean of response, both weighted and unweighted, and random forest variable importance is used for further analysis.

  - [p-value](https://en.wikipedia.org/wiki/P-value): The p-value tells you how likely it is to observe the data by chance alone. A small p-value (typically less than 0.05) suggests strong evidence against the null hypothesis, indicating that the observed effect is unlikely to be due to random chance.
  - [t-value](https://www.geeksforgeeks.org/t-test/): A t-test is a type of inferential statistic used to determine if there is a significant difference between the means of two groups, which may be related in certain features.
  - Generally a small p-value (less than 0.05) and large magnitude of t-value (>=2.00 and <= -2.00) is [considered good for confidence](https://www.allbusiness.com/barrons_dictionary/dictionary-t-value-4942040-1.html#:~:text=Generally%2C%20any%20t%2Dvalue%20greater,the%20coefficient%20as%20a%20predictor)
  - [Difference with Mean of Response](https://en.wikipedia.org/wiki/Variance_of_the_mean_and_predicted_responses): The difference with mean of response plots are used to find patterns in data. [For ex: If the age in the data increases, how the survival rate changes](https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/10/0/7). It is often analyzed using statistical tests to determine if the observed difference is statistically significant or if it could have occurred by chance.
  - Sometimes the population distribution is not uniform in different samples, in that case we take weighted mean of response. [Weighted mean of response gives better estimate of non uniformly spread data.](https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/4)
  - Examples of some good features:
    - ![Good_MOR_Plot_1](scripts/final/readme_images/Final_Good_MOR_Strikeout_Rate.png)
    - ![Good_MOR_Plot_2](scripts/final/readme_images/Final_Good_MOR_Strikeout_To_Walk_Ratio.png)
  - [Random Forest Variable Importance](https://en.wikipedia.org/wiki/Random_forest#Variable_importance): Random forests can be used to rank the importance of variables in a regression or classification problem in a natural way. The following technique was described in Breiman's original paper[9] and is implemented in the R package randomForest. Features which produce large values for this score are ranked as more important than features which produce small values. The statistical definition of the variable importance measure was given and analyzed by Zhu et al.
  - ![Continuous Feature Statistics](scripts/final/readme_images/Final_Continuous_Features_Statistics.png)

## 2.2 Pearson's Correlation for Continuous/ Continuous Predictors

- [Pearson's Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is a bivariate correlation between two sets of data. It is a linear correlation between two sets of data. The value of correlation lies in the range (-1,1) depicting the intensity of correlation, 1 being very high positive correlation and -1 being very high negative correlation.
- ![Pearson's Correlation Table](scripts/final/readme_images/Final_Correlation_Matrix.png)

## 2.3 Nominal/ Nominal Correlation

- Used [Cramer's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) and [Tschuprow's T](https://en.wikipedia.org/wiki/Tschuprow%27s_T) correlation matrix to find the correlation between nominal predictors.

## 2.3 Correlation Ratio

- Used [Correlation Ratio](https://en.wikipedia.org/wiki/Correlation_ratio) for the correlation between nominal and continuous predictors. The measure is defined as the ratio of two standard deviations representing these type of variation.

## 2.4 Brute Force Analysis

- A Continuous-Continuous Predictor Pairs Brute Force analysis is used to examine the relationship between two continuous predictors in a machine learning model. Two highly correlated features have their weights or coefficients divided and effectively do not contribute much to the model. Unlike a normal correlation ratio matrix, brute force analysis has mean of response analysis that helps to identity how to features together contribute in the mean of response. If two features together do not contribute effectively we discard one of those features.

# 3. Model Building

- I used different machine learning models in the pipeline to build and evaluate the models.

## 3.1 [Support vector machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine)

- The SVM, also known as Support Vector Machine (SVM), is a machine learning tool used for sorting things or making predictions. It works by drawing a line or boundary in a space with many dimensions to separate different groups or values. SVM is commonly used when the data can be easily divided by a straight line, but it can also handle more complicated cases with some tricks.

## 3.2 [Logistic Regression (LR)](https://en.wikipedia.org/wiki/Logistic_regression)

- Logistic regression is a statistical method used to predict the probability of an event occurring or a category being assigned. It models the relationship between the predictor variables and the outcome variable, typically in a binary classification setting. Its purpose is to aid in decision-making and understanding the impact of different factors on the outcome.

## 3.3 [K Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

- K-nearest neighbors (KNN) classifies new data points based on their similarity to nearby labeled data points. It determines the category of a new point by examining the categories of its k closest neighbors. KNN is a versatile algorithm used for classification and regression tasks, often in scenarios where the relationship between features and outcomes is complex or not well-defined.

## 3.4 [Random Forest Classifier (RFC)](https://en.wikipedia.org/wiki/Random_forest)

- Random Forest (RFC) is a machine learning ensemble algorithm that combines multiple decision trees to make predictions. Each tree is trained on a different subset of the data and features, and the final prediction is determined by aggregating the results from all the trees. RFC is known for its robustness, ability to handle complex relationships, and resistance to overfitting.

## 3.5 [Decision Tree Classifier (DTC)](https://en.wikipedia.org/wiki/Decision_tree)

- A decision tree is a flowchart-like model that represents possible decisions or outcomes and the potential consequences of each decision. It is a popular machine learning algorithm used for both classification and regression tasks. Decision trees use a tree-like structure where each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents a final decision or outcome. Decision trees are easy to understand and interpret, making them useful for decision-making and analyzing complex problems.

# 4. Model Evaluation

- I used different techniques like Precision, Recall, F1 Score, Accuracy, AUC Score to evaluate my models.
  - [Precision](https://www.analyticsvidhya.com/blog/2020/09/precision-recall-machine-learning/): In the simplest terms, Precision is the ratio between the True Positives and all the Positives.
  - ![Precision](scripts/final/readme_images/Precision.png)
  - [Recall](https://www.analyticsvidhya.com/blog/2020/09/precision-recall-machine-learning/): The recall is the measure of our model correctly identifying True Positives.
  - ![Recall](scripts/final/readme_images/Recall.png)
  - [F1 Score](https://en.wikipedia.org/wiki/F-score): F1 Score gives models accuracy in terms of precision and recall. It is harmonic mean of precision and recall.
  - [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision): Accuracy tells how close a given set of observations are to their true values. However, in case of binary classification it is not the only relevant metric to judge a model.
  - [AUC ROC](https://machine-learning.paperspace.com/wiki/auc-area-under-the-roc-curve): The ROC Curve measures how accurately the model can distinguish between two things (e.g. determine if the subject of an image is a dog or a cat). AUC measures the entire two-dimensional area underneath the ROC curve. This score gives us a good idea of how well the classifier will perform.

# 5. Conclusion

- I run 5 different models first, on all the features and then on different combination of features. I have included the model evaluation statistics and ROC curves below with all the features and then with the best combination of features after analyzing different combinations.
  - Model Evaluation with all the features
    ![Model Evaluation with all the features](scripts/final/readme_images/Final_Model_Evaluation_All_Features.png)
  - ROC Curve with all the features
    ![ROC Curve All Features](scripts/final/readme_images/Final_ROC_All_Features.png)
  - Model Evaluation with best features
    ![Model Evaluation with best features](scripts/final/readme_images/Final_Model_Evaluation_Best_Features.png)
  - ROC Curve with best features combination
    ![ROC Curve All Features](scripts/final/readme_images/Final_ROC_Best_Features.png)
- With all the features my best performing models are Decision Tree and Logistic Regression with AUC of 0.54 and 0.53
- With my best combination of features, the best performing models are SVC, KNN, Decision Tree wth AUC 0.54 and 0.53
