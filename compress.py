import pandas as pd
import os
import re

def compactData(year):
	# Concatenate all data files
	folder = "Data/" + str(year) + "-" + str(year + 1) + "/teamData/"
	directory = "/Users/johnlandy/Documents/Personal/CodingProjects/MarchMadness/" + folder
	files = os.listdir(directory)
	df = pd.concat((pd.read_csv(folder + f) for f in files), axis=1, sort=False)
	# Remove nondescriptive column headers
	df.columns = range(df.shape[1])
	# Remove columns by index
	if year == 2015 or year == 2017 or year == 2018:
		df = df.drop(df.columns[[0,3,4,16,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,60,
		61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,79,82,85,94,95,96,97,98,99,100,101,102,
		103,104,105,106,107,108,109,110,111,113,116,119]], axis=1)
	else:
		df = df.drop(df.columns[[0,3,4,16,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,60,
		61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,79,82,85,94,95,96,97,98,99,100,101,102,
		103,104,105,106,107,108,109,110,111,113,116,119]], axis=1)
	# Remove NCAA from team names
	for i in range(len(df[1])):
		df[1][i] = df[1][i].replace(" NCAA", "").replace("St.", "State")
		re.sub("[\(.[\)]", "", df[1][i])
	# Make all other elements floats
	for col in df:
		col = pd.to_numeric(col, downcast='float')
	# Export as csv
	df.to_csv(folder + "compact.csv")


def downloadTrainingData(folder, season):
	# Concatenate all data files
	df = pd.read_csv(folder + "training.csv")
	# Remove nondescriptive column headers
	df.columns = range(df.shape[1])
	# Remove columns by index
	if season == 2015 or season == 2018 or season == 2019:
		df = df.drop(df.columns[[0,1,2,8,9]], axis=1)
	elif season == 2016:
		df = df.drop(df.columns[[0,1,2,8,9,10,11,12]], axis=1)
	elif season == 2017:
		df = df.drop(df.columns[[0,6,7,8,9,10,11,12,13,14,15,16,17,18,19]], axis=1)
	# Export as csv
	df.to_csv(folder + "training.csv")


def main():
	# Create compact team vectors
	seasons = [2015, 2016, 2017, 2018, 2019]
	# Compress downloaded data and compile training data
	for year in seasons:
		compactData(year)
		#downloadTrainingData("Data/" + str(year) + "-" + str(year + 1) + "/", year)


if __name__ == "__main__":
	main()
