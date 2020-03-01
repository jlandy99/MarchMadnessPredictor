#!/bin/bash

# Remove compact data set for each folder in Data
list="Data/2015-2016/teamData/compact.csv
Data/2016-2017/teamData/compact.csv
Data/2017-2018/teamData/compact.csv
Data/2018-2019/teamData/compact.csv
Data/2019-2020/teamData/compact.csv"

for y in $list
do
	rm -rf $y
done

# Run the data compression file
python3 compress.py
# Run the training file
python3 train.py
