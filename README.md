# MarchMadnessPredictor
This project is a machine learning classifier trained on about 5 full seasons of NCAA basketball data. It uses pandas to parse and trim data, and sklearn to build and train models. Predictions for 2020 March Madness will be pushed below once model is finalized and hyperparameters are fine-tuned.

## Approach
To create this project, I read up on several other similar projects, and used that reading, as well as some of the material I learned in school, to try to create my own classifier. I will walk through my procedure below:

## Procedure
#### Data Collection and Parsing
The first challenge in this project was obtaining sufficient data. There are tons of NCAA data out there, but finding well-formatted, consistent, precise data makes a huge difference. I got most of my data from sports-reference.com, from which I found yearly team data that I put together. I then found an archive of matchup data (team A, team B, site, score), and used these two datasets together as training data. Each feature in my training data was a vector of all the statistics/attributes I was keeping track of for a team, but in this case, to make it specific to a matchup, I subtracted one from the other. I also added in an attribute for location of the game. The label was simply a 0 or 1 for a win or loss. My model was built on regression, but for the training data, I had all labels 0 or 1. 

I built my training data from the past 5 years of NCAA games. Since I obtained my matchup data and team data from different sources, the team names were inconsistent, and many of them didn't match up, but after changing some common abbreviations, I had over 30,000 training points, and figured that would be sufficient.

#### Model Selection
I then began looking at different models to use. I started with a simple Linear SVC, which performed pretty poorly, at around 56.5% accuracy on k-fold cross validation. I then incorporated a feature selection method from sklearn called mutual_info_classif, which essentially measures dependency between random variables to perform feature selection. I used this because I was using a ton of statistics that I was able to find online, and I wanted to simplify my model (I trimmed from about 65 features to 32). I messed around with a few other classifiers, but the one that seemed to work the best was Gradient Boosting Regressor, similar to AdaBoost. Tweaking the parameters, I managed to get my accuracy on k-fold cross validation up to about 77.5%. Some of the plots for my hyperparameter tuning can be found in the Plots folder.

Note: My model performs regression, and thus returns float values near the range [0, 1]. For k-fold cross validation, I made my label space {0, 1} by making any point >= 0.5 move to 1 and any other point move to 0.

#### Prediction
I then prepared my model to work on tournament data. I did this by allowing a tournament bracket from year x to be tested using all regular season and tournament data from every year except x. While it could be argued that it's advantageous to use March Madness data only, I chose to use all regular season data because I valued the magnitudes higher quantity over the slightly better precision.

For test.py, I incorporated a few new features into my model. For my training data, I mapped any value >= 0.5 to 1 and any other value to 0. However, in testing tournament matchups, I preferred to run them many times and take the average. Using a regression model as opposed to classification allows me to incorporate an idea of strength of win/loss as opposed to just win/loss, which should help predictions. I ran each matchup a certain amount of times, adding up all of the float values I got (clipping between 0 and 1 when necessary) and then took the average. This was especially helpful because it gave me more verbose output, and I could quantify it in terms of a percent chance that Team A wins over Team B.

###### Randomness
I also noticed that upon predicting my first tournaments, the predictions were largely scratch. I wanted to add in more chance to my model, due to the high nature of upsets in this type of tournament. I chose to implement a couple of strategies to achieve this. First, I varied hyperparameters around their optimal values when building my model per matchup. Thus, the average of several iterations of a single matchup would be composed of regressors with different tuning.

###### Formatting Predictions
The last thing I did was functionalized formatting brackets correctly. This was largely unnecessary but it is nice to be able to see a bracket without having to write anything down.

## Improvements
There are a bunch of things that I would still like to add to this project before March Madness comes around and I have to finalize my model. 

#### Statistics/Metrics
I have included a bunch of statistics and metrics as well as trimming them down based on correlation, but I haven't done any additional research on some more unique statistics to use about a team which may be less quantifiable, but helpful to incorporate, like coaching strength, chemistry, etc.

It would also be interesting to try to incorporate a feature which describes a team's momentum. This would be a good way to not just look at records and averages coming into the tournament, but also trends.

#### Historical Data
I have not yet incorporated historical tournament data into my model. This is not going to be a simple addition, but it definitely would be interesting to see if adding this could increase my accuracy.
