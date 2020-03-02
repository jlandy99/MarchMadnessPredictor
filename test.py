import math
import numpy as np
import pandas as pd
from sklearn import feature_selection, preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from train import buildTeamVectors, formatTrainingData


# Function which takes in a combined vector from 2 teams and predicts the winner
def testGames(clf, X_test):
	# return result of predict function, clipped to between 0 and 1
	return np.clip(clf.predict(X_test), 0, 1)


# Used for truncating percentages to the 10th place
def truncate(n, decimals=1):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# Helper function used by updateMatchups when round == 0
def updateRound1(data, testingFolder):
	# For each slash in round 1 data, find corresponding team in data and write them in
	df = pd.read_csv(testingFolder + "round1.csv")
	for index, row in df.iterrows():
		# Find teams with slashes and pick the winners
		x = []
		if "/" in row[0]:
			x = row[0].split("/")
			# Pick the winning team of the 2
			for elt in data:
				if elt == x[0] or elt == x[1]:
					row[0] = elt
		if "/" in row[1]:
			x = row[1].split("/")
			# Pick the winning team of the 2
			for elt in data:
				if elt == x[0] or elt == x[1]:
					row[1] = elt
	# Resave csv file
	df = pd.DataFrame(df, columns = ['team1', 'team2'])
	df.to_csv(testingFolder + "round1.csv", index=False)


# Helper function used by updateMatchups to write results csv files per round
def writeResults(year, percents, round):
	testingFolder = "Data/" + str(year) + "-" + str(year + 1) + "/testingData/"
	# Save as csv file
	df = pd.DataFrame(percents, columns = ['team', 'percentage'])
	df.to_csv(testingFolder + "results" + str(round) + ".csv", index=False)


# Function which serves to update/create csv files accordingly based on round and results
def updateMatchups(year, results, round):
	# Find our folder and then open the csv file of the just completed round
	testingFolder = "Data/" + str(year) + "-" + str(year + 1) + "/testingData/"
	df = pd.read_csv(testingFolder + "round" + str(round) + ".csv")
	# Dictionary to keep track of percentages
	percents = []
	# Check round number; special case if 0
	if round == 0:
		# If our round is 0, we want to look through the partially filled round1.csv
		# And change all of the slashes to the teams that won the play-in games
		# Go through each row of round 0 data
		dict = {}
		count = 0
		for index, row in df.iterrows():
			team1 = row[0]
			team2 = row[1]
			# Check results (more 1s means team 1 wins)
			if results[count] > 0.5:
				dict[team1] = results[count] * 100
				percents.append([team1, truncate(results[count] * 100)])
			else:
				dict[team2] = (1 - results[count]) * 100
				percents.append([team2, truncate((1 - results[count]) * 100)])
			count += 1
		# Now that we've created our dictionary, we can update round1.csv
		updateRound1(dict, testingFolder)
	else:
		# Otherwise, we want to create a csv for the next round and write it to
		# an accessible location for our next iterations
		# Iterate through our results and pin adjacent winners with each other
		count = 0
		team1s = []
		team2s = []
		for index, row in df.iterrows():
			team1 = row[0]
			team2 = row[1]
			# Check results (more 1s means team 1 wins)
			if results[count] > 0.5:
				percents.append([team1, truncate(results[count] * 100)])
				if count % 2 == 0:
					team1s.append(team1)
				else:
					team2s.append(team1)
			else:
				percents.append([team2, truncate((1 - results[count]) * 100)])
				if count % 2 == 0:
					team1s.append(team2)
				else:
					team2s.append(team2)
			count += 1
		# Check if we've found a champion
		if len(team1s) + len(team2s) == 1:
			print("CHAMPION FOUND!")
			# If we have, write results but do not proceed
			writeResults(year, percents, round)
			return
		# Now, we should have a 2xn list to put into a new dataframe
		dict = {"team1": team1s, "team2": team2s}
		df2 = pd.DataFrame(dict, columns = ['team1', 'team2'])
		# Write it to a new csv file for the next round
		df2.to_csv(testingFolder + "round" + str(round + 1) + ".csv", index=False)
	# Write results data from each round into results.csv
	writeResults(year, percents, round)


# Function which runs round 0
def roundSim(year, games, round, iterations):
	# Iterate 10 times over to predict results
	results = np.zeros(games)
	# Operations we only need to perform once
	# Building our team vectors
	dict = buildTeamVectors(seasons=[year])
	# Iterate over the dataset multiple times and average
	for i in range(iterations):
		# Train our classifier (also returns selector for features on testing data)
		clf, selector = trainClassifier(year)
		# Find the round data csv files in the corresponding folder
		testingFolder = "Data/" + str(year) + "-" + str(year + 1) + "/testingData/"
		trainingFolder = "Data/" + str(year) + "-" + str(year + 1) + "/teamData/"
		# First, run round 0 (play in games)
		X_test = np.zeros((games, 65))
		# Skim csv file and simulate all games (4)
		df = pd.read_csv(testingFolder + "round" + str(round) + ".csv")
		count = 0
		for index, row in df.iterrows():
			team1 = row[0]
			team2 = row[1]
			# Combine the teams into one vector
			vec1 = dict.get(team1 + str(year), None)
			vec2 = dict.get(team2 + str(year), None)
			if vec1 is None:
				print("Error: Couldn't find " + team1)
				return
			elif vec2 is None:
				print("Error: Couldn't find " + team2)
				return
			# Now we can proceed to combine and test the vectors
			combined = []
			for elt in range(len(vec1)):
				combined.append(float(vec1[elt]) - float(vec2[elt]))
			# Add in neutral game site
			combined.append(0)
			# Append to X_test
			X_test[count] = combined
			count += 1
		# Normalize X_test
		X_test = preprocessing.normalize(X_test)
		# Use our selector to trim X_test (based on feature correlation)
		X_test = selector.transform(X_test)
		# Test the match up with our trained clf object
		y_pred = testGames(clf, X_test)
		results = results + y_pred
		print(y_pred)
	# Normalize results to fractions
	results = results / iterations
	# Update csv files accordingly
	updateMatchups(year, results, round)


# Function that handles the training of a classifier
def trainClassifier(year):
	# Use optimal parameters from train.py (vary around optimal values for chance)
	estimators = math.floor(np.random.uniform(70, 110, 1))
	max_depth = math.floor(np.random.uniform(4, 5, 1))
	max_features = math.floor(np.random.uniform(10, 30, 1))
	learning_rate = np.random.uniform(5, 20, 1)/float(100)
	params = {"n_estimators": estimators, "max_depth": max_depth,
				"max_features": max_features, "learning_rate": learning_rate}
	# Create a Gradient Boosting Classifier from these parameters
	clf = GradientBoostingRegressor(**params)
	# Run on all training data except current year
	seasons = [2015, 2016, 2017, 2018, 2019]
	seasons.remove(year)
	# Build team vectors and format training data
	data = buildTeamVectors(seasons=seasons)
	X_train, y_train = formatTrainingData(data, seasons=seasons)
	# Normalize X_train
	X_train = preprocessing.normalize(X_train)
	# Remove columns with low correlation to label outcome
	selector = feature_selection.SelectPercentile(feature_selection.mutual_info_classif, percentile=50).fit(X_train, y_train)
	X_train = selector.transform(X_train)
	# Train our clf on the training data
	clf.fit(X_train, y_train)
	# Return the clf object
	return clf, selector


# Helper to make all team names equal length strings
def makeNchars(word, n):
	if len(word) > n:
		return word[:n]
	else:
		while len(word) != n:
			word += " "
		return word


# Helper function to buildBracket for round 0
def buildRound0(folder, year):
	# Create text file with w+ flag (first round running)
	f = open("Data/" + str(year) + "-" + str(year + 1) + "/bracket.txt", "w+")
	# Write the sub-heading play-in games
	f.write("PLAY-IN GAMES\n\n")
	# Row 1: All team 1s in round 0
	df = pd.read_csv(folder + "round0.csv")
	for index, row in df.iterrows():
		# Write the first team (up to 10 chars) followed by 4 tabs
		word = str(row[0])
		f.write(makeNchars(word, 12))
		f.write("\t\t\t\t")
	f.write("\n")
	# Row 2: All results of games (results and round csv files are in same order)
	df = pd.read_csv(folder + "results0.csv")
	# Iterate over each row in the results file
	for index, row in df.iterrows():
		f.write("\t\t")
		f.write(str(row[1]))
		f.write("% ")
		winner = str(row[0])
		f.write(makeNchars(winner, 12))
		f.write("\t")
	f.write("\n")
	# Row 2: All team 2s in round 0
	df = pd.read_csv(folder + "round0.csv")
	for index, row in df.iterrows():
		# Write the first team (up to 10 chars) followed by 4 tabs
		word = str(row[1])
		f.write(makeNchars(word, 12))
		f.write("\t\t\t\t")
	f.write("\n\n")
	f.close()


# Helper which puts the winners of each round into lists so that it is easier
# to correctly format a bracket from scratch
def listRounds(year, start, stop):
	# Save each set of teams to a 3d list
	rounds = []
	list = []
	folder = "Data/" + str(year) + "-" + str(year + 1) + "/testingData/"
	# Do first round first, as no percentages are needed
	counter = 16
	df = pd.read_csv(folder + "round" + str(start) + ".csv")
	for i in range(counter):
		leftMatch = df.loc[i, :]
		rightMatch = df.loc[i + counter, :]
		# Pair up as they will appear on a line of the bracket
		temp1 = [makeNchars(leftMatch["team1"], 12), makeNchars(rightMatch["team1"], 12)]
		temp2 = [makeNchars(leftMatch["team2"], 12), makeNchars(rightMatch["team2"], 12)]
		list.append(temp1)
		list.append(temp2)
	# Add round 1 to the rounds list
	rounds.append(list)
	# Iterate on range 1 to 7 for results files
	for round in range(start, stop - 1):
		# Reset the current list and read in the csv file for the results
		list = []
		df = pd.read_csv(folder + "results" + str(round) + ".csv")
		# Iterate through all games
		for i in range(counter):
			# Find winners on each row
			leftWinner = df.loc[i, :]
			rightWinner = df.loc[i + counter, :]
			# Winner will be in the form teamA,xx.x
			left = str(leftWinner["percentage"]) + "% " + makeNchars(leftWinner["team"], 12)
			right = str(rightWinner["percentage"]) + "% " + makeNchars(rightWinner["team"], 12)
			list.append([left, right])
		# Narrow the counter for the following round
		counter = int(counter / 2)
		# Append the current round to the full list of results before resetting
		rounds.append(list)
	# Handle final round (different formatting)
	df = pd.read_csv(folder + "results" + str(stop - 1) + ".csv")
	# Add the winner and percentage as a 1 element array directly to rounds
	winner = df.loc[0, :]
	rounds.append([str(winner["percentage"]) + "% " + makeNchars(winner["team"], 12)])
	# Return the 3-d list of rounds and results
	return rounds


# Tool which actually handles the formatting of a bracket into a text file
def buildBracket(year):
	folder = "Data/" + str(year) + "-" + str(year + 1) + "/testingData/"
	# For round 0 it is easy, we will simply use the top three lines
	buildRound0(folder, year)
	# Get individual lists of rounds
	rounds = listRounds(year, 1, 7)
	# Array holding the sequence of rounds[i] to use working downwards (note 6 = 6 and 7)
	order = [0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
			0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0]
	# Array of tabs to separate on based on round (by index - 1)
	pretabs = [0, 2, 5, 8, 10, 12]
	midtabs = [31, 25, 19, 13, 9, 0]
	# Array which keeps track of the index within each rounds[i] to use next in below loop
	indeces = [0, 0, 0, 0, 0]
	# Open our file to write to
	f = open("Data/" + str(year) + "-" + str(year + 1) + "/bracket.txt", "a")
	# Write a sub-header for the regular section of the tournament
	f.write("TOURNAMENT GAMES\n\n")
	# Iterate through the order array and pull from results to format
	for i in range(len(order)):
		# Write the "next" row of rounds[i]
		round = order[i]
		# Special case if round == 5
		if round == 5:
			for i in range(pretabs[round]):
				f.write("\t")
			# Semifinalist 1
			f.write(rounds[round][0][0])
			f.write("\t")
			# Champion
			f.write(rounds[round + 1][0])
			f.write("\t")
			# Semifinalist 2
			f.write(rounds[round][0][1])
		else:
			for i in range(pretabs[round]):
				f.write("\t")
			# Left team
			f.write(rounds[round][indeces[round]][0])
			for i in range(midtabs[round]):
				f.write("\t")
			# Right team
			f.write(rounds[round][indeces[round]][1])
			# Increment the index to use next for the given round (not needed for round == 5)
			indeces[round] += 1
		# increment to the next line
		f.write("\n")
	# Save our file
	f.close()


# Interactive function which will run a bracket prediction for a given year
# and compare results to actual if they exist
def main():
	# Repeat until they exit
	while True:
		# Get the year
		year = int(input("Enter a year (when season started): "))
		# Decide on number of iterations to average
		iterations = 10
		# Run round 0
		roundSim(year, 4, 0, iterations)
		# Run the rest of the rounds
		combVectors = 32
		for i in range(1, 7):
			print("Running Round " + i + "...\n")
			roundSim(year, combVectors, i, iterations)
			combVectors = int(combVectors / 2)
		# Build and format bracket
		buildBracket(year)
		# Print formatting details
		print("Note: bracket.txt is best viewed in Microsoft Word")
		# Ask if they'd like to loop again
		again = input("Would you like to run again (y/n)? ")
		if again == "n":
			break
	print("Thanks for using this classifier..")


if __name__ == "__main__":
	main()
