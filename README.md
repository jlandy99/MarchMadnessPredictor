# MarchMadnessPredictor
This project is a machine learning classifier trained on about 5 full seasons of NCAA basketball data. It uses pandas to parse and trim data, and sklearn to build and train models. Predictions for 2020 March Madness will be pushed below once model is finalized and hyperparameters are fine-tuned.

## Approach
To create this project, I read up on several other similar projects, and used that reading, as well as some of the material I learned in school, to try to create my own classifier. I will walk through my procedure below:

## Procedure
The first challenge in this project was obtaining sufficient data. There are tons of NCAA data out there, but finding well-formatted, consistent, precise data makes a huge difference. I got most of my data from sports-reference.com, from which I found yearly team data that I put together. I then found an archive of matchup data (team A, team B, site, score), and used these two datasets together as training data. Each feature in my training data was a vector of all the statistics/attributes I was keeping track of for a team, but in this case, to make it specific to a matchup, I subtracted one from the other. I also added in an attribute for location of the game. The label was simply a 0 or 1 for a win or loss. I could have maybe added in a score difference, and run more of a regression problem, but I wanted to get something working before I made it too complex.

I built my training data from the past 5 years of NCAA games. Since I obtained my matchup data and team data from different sources, the team names were inconsistent, and many of them didn't match up, but after changing some common abbreviations, I had over 30,000 training points, and figured that would be sufficient. I then began looking at different models to use. I started with a simple Linear SVC, which performed pretty poorly, at around 57.5% accuracy on k-fold cross validation. I then incorporated a feature selection method from sklearn called mutual_info_classif, which essentially measures dependency between random variables to perform feature selection. I used this because I was using a ton of statistics that I was able to find online, and I wanted to simplify my model. I messed around with a few other classifiers, but the one that seemed to work the best was Gradient Boosted Decision Trees, similar to AdaBoost. Tweaking the parameters, I managed to get my accuracy on k-fold cross validation up to about 77.5%. Some of the plots for my hyperparameter tuning can be found in the Plots folder.

## Improvements
There are a bunch of things that I would still like to add to this project before March Madness comes around and I have to finalize my model. 

#### Statistics/Metrics
I have included a bunch of statistics and metrics as well as trimming them down based on correlation, but I haven't done any additional research on some more unique statistics to use about a team which may be less quantifiable, but helpful to incorporate, like coaching strength, chemistry, etc.

It would also be interesting to try to incorporate a feature which describes a team's momentum. This would be a good way to not just look at records and averages coming into the tournament, but also trends.

#### Historical Data
I have not yet incorporated historical tournament data into my model. This is not going to be a simple addition, but it definitely would be interesting to see if adding this could increase my accuracy.
