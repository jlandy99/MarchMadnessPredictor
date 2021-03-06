import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, feature_selection, preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


# Builds team vectors out of csv file
def buildTeamVectors(seasons=[2015, 2016, 2017, 2018, 2019]):
	dict = {}
	sizes = set()
	# Read in our compacted dataset for the current season
	for year in seasons:
		df = pd.read_csv("Data/" + str(year) + "-" + str(year + 1) + "/teamData/compact.csv")
		# Ignore the header row
		for index, row in df.iterrows():
			if row[1] != "School":
				dict[row[1] + str(year)] = list(row[2:])
	return dict


# Trim training data
def trimData(str):
	str = str.replace("St.", "State").replace("Mich.", "Michigan")
	str = str.replace("So.", "Southern").replace("Ill.", "Illinois")
	str = str.replace("Ark.", "Arkansas").replace("Tenn.", "Tennessee")
	str = str.replace("Ky.", "Kentucky").replace("Val.", "Valley State")
	str = str.replace("Ala.", "Alabama").replace("Ga.", "Georgia")
	str = str.replace("Mo.", "Missouri").replace("Colo.", "Colorado")
	str = str.replace("Caro.", "Carolina").replace("N.C.", "North Carolina")
	str = str.replace("Miss.", "Mississippi").replace("Ariz.", "Arizona")
	str = str.replace("Conn.", "Connecticut").replace("U.", "University")
	str = str.replace("Fla.", "Florida").replace("UC ", "UC-")
	str = str.replace("UT ", "UT-").replace("Wash.", "Washington")
	str = str.replace("Ole Miss", "Mississippi").replace("La.", "Louisiana")
	str = str.replace("UT-Martin", "Tennessee-Martin").replace("Seattle U", "Seattle")
	str = str.replace("UT-Arlington", "Texas-Arlington").replace("Ind.", "Indiana")
	str = str.replace("East.", "Eastern").replace("Wis.", "Wisconsin")
	str = str.replace("Tex.", "Texas").replace("Atl.", "Atlanta")
	str = str.replace("Int'l", "International").replace("Christ.", "Christian")
	str = str.replace("Alas.", "Alaska").replace("N.M.", "New Mexico")
	str = str.replace("Okla.", "Oklahoma").replace("Bapt.", "Baptist")
	str = str.replace("Bowling Green", "Bowling Green State")
	str = str.replace("LSU", "Louisiana State").replace("USC", "Southern California")
	str = str.replace("BYU", "Brigham Young").replace("Saint Mary's", "Saint Mary's (CA)")
	return str


# Format our whole X_train and y_train from our dictionary of team vectors
def formatTrainingData(data, seasons=[2015, 2016, 2017, 2018, 2019]):
	# Find out size of X_train, Y_train
	size = 0
	for year in seasons:
		df = pd.read_csv("Data/" + str(year) + "-" + str(year + 1) + "/training.csv")
		for index, row in df.iterrows():
			team1 = trimData(row[1])
			team2 = trimData(row[2])
			vec1 = data.get(team1 + str(year), None)
			vec2 = data.get(team2 + str(year), None)
			if vec1 is not None and vec2 is not None:
				size += 1
	# Set size
	X_train = np.zeros((size, 65))
	y_train = np.zeros((size))
	unmatched = set()
	count = 0
	count1 = count2 = 0
	# Iterate through our training.csv file to build to our X_train and Y_train
	for year in seasons:
		df = pd.read_csv("Data/" + str(year) + "-" + str(year + 1) + "/training.csv")
		# Iterate by row and formulate training vector
		for index, row in df.iterrows():
			# Find team vectors in dictionary (data)
			team1 = trimData(row[1])
			team2 = trimData(row[2])
			vec1 = data.get(team1 + str(year), None)
			vec2 = data.get(team2 + str(year), None)
			if vec1 is not None and vec2 is not None:
				# Build our composite vector
				combined = []
				for elt in range(len(vec1)):
					combined.append(float(vec1[elt]) - float(vec2[elt]))
				# Append a row for the site of the game
				# We will use 1 if vec 1 home (H), 0 if neutral (N), and -1 if vec 2 home (V)
				if row[3] == "V":
					combined.append(-1)
				elif row[3] == "H":
					combined.append(1)
				else:
					combined.append(0)
				# Find who won (1 for team 1, 0 for team 2)
				result = 1 if row[4] - row[5] > 0 else 0
				# Append this vector to our X_train, and now the result to our y_train
				X_train[count] = combined
				y_train[count] = result
				count += 1
				count1 += 1
			else:
				# Print out the team names that didn't register so we can build our alias
				if vec1 is None:
					unmatched.add(team1)
				if vec2 is None:
					unmatched.add(team2)
				count2 += 1
		# Write all unmatched teams to file
		f = open("unmatchedTeams.txt", "w")
		for team in unmatched:
			f.write(team)
			f.write("\n")
	# Return our training data
	return X_train, y_train


# Runs k-fold cross validation on the training data for the year to build a model
def crossValidate(clf, X, y, k, percent=50):
	# Keep track of the performance of the model on each fold in the scores array
	scores = []
	# Create the object to split the data
	skf = StratifiedKFold(n_splits=k)
	count = 1
	# Iterate through the training and testing data from each of the k-fold splits
	for train_index, test_index in skf.split(X, y):
		# Get our training and testing data to use from the split function
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# Remove columns with low correlation to label outcome
		selector = feature_selection.SelectPercentile(feature_selection.mutual_info_classif, percentile=percent).fit(X_train, y_train)
		X_train = selector.transform(X_train)
		# Note the columns we remove must be extracted from X_test as well here
		X_test = selector.transform(X_test)
		# Normalize data
		X_train = preprocessing.normalize(X_train)
		X_test = preprocessing.normalize(X_test)
		# Fit based on the training normalized data
		clf.fit(X_train, y_train)
		# Update the scores array with the performance on the testing data
		y_pred = clf.predict(X_test)
		# Move all values to binary classifications
		y_pred[y_pred >= 0.5] = 1
		y_pred[y_pred < 0.5] = 0
		# Our function for the prediction will vary depending on the metric
		accuracy = metrics.accuracy_score(y_test, y_pred)
		print(accuracy)
		scores.append(accuracy)
		count += 1
	# Return the average performance across all fold splits.
	return np.array(scores).mean()


# Helper to plot results
def plotter(accuracy, variable):
	plt.plot(variable, accuracy)
	plt.xlabel("percentile feature selection")
	plt.ylabel("Accuracy")
	plt.title("percentile.png")
	plt.savefig("percentile.png")
	plt.close()


# Tunes hyperparameters within the Gradient Boost Classifier
def tuneHyperParameters(X_train, y_train):
	# Find optimal learning Rate
	percentiles = [50]
	accuracy_scores = []
	for i in percentiles:
		# Now, run k-fold cross validation on the training vectors
		params = {"max_depth": 4, "max_features": 26}
		clf = GradientBoostingRegressor(**params)
		accuracy = crossValidate(clf, X_train, y_train, 5, percent=i)
		print("Percentile: " + str(i) + ", Accuracy: " + str(100 * accuracy))
		accuracy_scores.append(accuracy)
	# Plot data
	plotter(accuracy_scores, percentiles)


def main():
	# Begin by compiling the training vectors for each year
	data = buildTeamVectors()
	# Format our training dataset
	X_train, y_train = formatTrainingData(data)
	# Tune our hyperparameters
	tuneHyperParameters(X_train, y_train)


if __name__ == "__main__":
	main()
