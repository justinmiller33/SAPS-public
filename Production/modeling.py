# Modeling.py
# Static File intended to research possible policies for use in simulatioin.py
# Does not modify data or simulate live policy enactment

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
from matplotlib import pyplot as plt
import random
import math
import warnings
import time
import re
import json


"""
DISCLAIMER:
This class is not complete.
IN THE WORKS: implementation of ML models from old scripts.
This class is solely for policy exploration.
Simulation and Trading is carried out by other .py files.

---------
OVERVIEW:
---------
Creating a class to generalize a pipeline for different models
Includes:
0. Initializing Data Stream
1. Creating assertion functions for robustness
2. Defining 'profit' and cleaning data for ML algorithms
3. Gathering supplemental data for feature engineering 
4. Basic ML Models and Policies
5. Ensembling and Running Retrospective Simulations

Notes:
- Time descriptions |t| are estimates and may vary based on the data
- CALLS details custom methods called in the function
- CALLS ignores __init__ and second-level functions
- REQS  details methods that must be called before the function can be run
- REQS ignores __init__

----------
DIRECTORY:
----------

Properties:
data = complete dictionary of data (see kaggle for download)
subList = list of all subreddits included in the data
subTemp = current subreddit of interest
finType = type of financial data, intraday or interday
profitDict = dictionary to assign profits to
missedDict = dictionary to list ignored keys to


Methods:

0.0 __init__
(pathToJson) : |4 Seconds / 100 MB|
initializing data through the post df, financial data fData, and post/fData keys with data.keys
CALLS: loadJson
REQS: n/a

0.1 loadJson
(pathToJson) : |4 Seconds / 100 MB|
loading json file from local path and reserializing some pd series and dataframes
CALLS: n/a
REQS: n/a

0.2 initProfitDicts
() : |insignificant|
initializing (or resetting) profit and missed dicts to handle profit assignments
CALLS: n/a
REQS: n/a

1.0 assertInter
() : |insignificant|
checking class property finType to ensure it is 'inter'
REQS: n/a

1.1 assertIntra
() : |insignificant|
checking class property finType to ensure it is 'intra'
REQS: n/a

2.0 holdForTime
(days) : | <1 Second |
Calculating the profitablity of an n-day hold policy
REQS: initProfitDicts

2.1 cutMissedProfits
() : | insignificant |
Trimming raw postData to rows with profits adherent to the current policy 
REQS: holdForTime

2.2 removeTickersFromText
() : | insignificant |
Trimming ticker name from title and text to avoid confounding for naive bayes
REQS: cutMissedProfits


"""

class pipeline:

    """
    0.0-2
    Section for Initializing Class and Loading Data
    """
    
    # Initializing the pipeline to allow for analysis on subsets of the scraped data
    def __init__(self,pathToJson,**kwargs):
        self.data = pipeline.loadJson(pathToJson)
        self.subList = list(self.data.keys())
        self.subTemp = None
        self.finType = 'inter'
        self.profitDict = {}
        self.missedDict = {}
        self.info = {}
        
    # Default function to open master json file
    # Also processes certain data into pandas dataframes and series
    def loadJson(pathToJson):

        # Load data as pandas df
        data = pd.read_json(pathToJson)

        # Convert back to original dict
        data = data.to_dict()

        # Loop through each subreddit
        for subreddit in data.keys():

            # Convert post data into labeled dataframe
            data[subreddit]['raw']['postData'] = pd.DataFrame(data[subreddit]['raw']['postData'], columns = ["ticker","title","text","flair","unix"])

            # For each financial data type
            for finType in ["inter","intra"]:

                # Get the data and transform it into a pandas time series
                for key in list(data[subreddit]['raw'][finType].keys()):

                    newData = np.array(data[subreddit]['raw'][finType][key])
                    newData = pd.Series(data = newData[:,1], index = newData[:,0])
                    data[subreddit]['raw'][finType][key] = newData
                    
        return data

    # Initialize (or reset) profit dictionaries to assign profits for analysis
    def initProfitDicts(self):
        for sub in self.subList:
            self.profitDict[sub] = {}
            self.missedDict[sub] = []

    """
    1.0-1.1
    Minor Functions for Checks and Assertions
    """
    def assertInter(self):
        if self.finType != "inter":
            raise Exception("This Function can only be called with .finType = 'inter'")

    def assertIntra(self):
        if self.finType != "intra":
            raise Exception("This Function can only be called with .finType = 'intra'")

    """
    2.0-2
    Assigning Profits and Reorganizing within postData
    """

    # Assigning profits via hold time
    def holdForTime(self,days):

        # Assert that self.finType is inter
        self.assertInter()

        # Updating self.info with number of days chosen
        self.info['daysToHold'] = days

        # For each subreddit
        for sub in self.subList:

            # For each key in the data
            for key in self.data[sub]['md'][self.finType]['keys']:

                # Get the profit time series for that post
                dataTemp = self.data[sub]['raw'][self.finType][key]

                # Try-except for recent posts that don't have that many days
                try:
                    # Get unique timestamps (only 1 per day)
                    dayListUnix = np.unique(dataTemp.index)
                    # Find the target stamp
                    targetUnix = dayListUnix[days]

                    # Profit is the last index for that day, assuming 4:00 sell
                    profit = 1 + dataTemp[targetUnix].iloc[-1]
                    self.profitDict[sub][key] = profit

                except:
                    # If there isn't enough data for a n-day hold, mark it as missed
                    self.missedDict[sub].append(key)

    # Function to cut postData to dataframe with profits from time
    # NOTE... can only run once per test without reloading json
    def cutMissedProfits(self):

        # Assert inter
        self.assertInter()
        
        # For each subreddit
        for sub in self.subList:

            # Convert indices to str to match keys to raw postData df
            saps.data[sub]['raw']['postData'].index = saps.data[sub]['raw']['postData'].index.astype(str)

            # Make new df keeping only posts with profits
            saps.data[sub]['raw']['postData'] = saps.data[sub]['raw']['postData'].loc[list(saps.profitDict[sub].keys())]


    # Adding profits to dataframe for easier manipulation
    def appendProfitsToDf(self):

        # Assert inter
        self.assertInter()

        # For each subreddit
        for sub in self.subList:

            # Create a new column with profit data
            saps.data[sub]['raw']['postData']['profit'] = list(saps.profitDict[sub].values())

    """
    3.0-1
    Supplementing postData and Removing Confounding Vars for ML algos
    """

    # Function to remove tickers from text to reduce confounding in naive bayes
    # Not possible to remove everything, will experience time-relevancy bias
    def removeTickersFromText(self):

        # For each subreddit
        for sub in self.subList:

            # Assign temporary dataframe
            dfTemp = saps.data[sub]['raw']['postData']

            # For each row in the df
            for rowNum in range(len(dfTemp)):

                # Removve the ticker from the text and/or title
                dfTemp['text'][rowNum] = dfTemp['text'][rowNum].replace(dfTemp.ticker[rowNum],"")
                dfTemp['title'][rowNum] = dfTemp['title'][rowNum].replace(dfTemp.ticker[rowNum],"")
            

    def simulateProfits(self):

        # Which sub to run
        for sub in self.subList:

            # Saving df
            df = self.data[sub]['raw']['postData']

            # Starting money and keys
            bank = 1000
            portfolio = 1000
            origInvest = 0
            investList = list()
            holdingKeys = list()

            # Assigning Start Time
            # time1 = np.min(df.unix)-10
            # Start time at start of volatility 3/11/2020
            time1 = df.unix[df.unix > 1583884800][-1] - 1
            time2 = time1 + 3600
            
            endTime = 1600500000
            print("started " + sub)
            while time2 < endTime:

                # Getting range of keys between those times (reversed to start with first buy
                keysInRange = list(df.unix[time1 < df.unix][time2 > df.unix].index)[::-1]

                # Adding those to holdingKeys
                holdingKeys = holdingKeys + keysInRange
                
                # Subtracting from bank ($1 EACH)
                bank = bank - len(keysInRange)
                investList = investList + len(holdingKeys) * [portfolio/1000]

                #origInvest = origInvest + len(keysInRange)
                
                
                # Starting loop to check for keys who's sell time has come
                timeToHold = (self.info['daysToHold'] + 2*self.info['daysToHold']/5) * 86400
                checkedHold = False
                
                while not checkedHold:

                    if not len(holdingKeys):
                
                        print(bank)
                        time2 = endTime
                        break
                        
                    keyToCheck = holdingKeys[0]

                    unixOfKey = df.unix[keyToCheck]
                    
                    if time2 > unixOfKey + timeToHold:

                        bank = bank + investList[0] * df.profit[keyToCheck]
                        portfolio = portfolio - investList[0] + investList[0] * df.profit[keyToCheck]
                        print(portfolio)
                        #origInvest = origInvest-1
                        
                        del holdingKeys[0]
                        del investList[0]
                        
                        continue
                    
                    else:
                        checkedHold = True


                time1 = time2
                time2 = time2 + 3600
                    
# Input local data path
pathToJson = "/home/justinmiller/devel/SAPS-public/Data/saps.json"

# Initialize pipeline with json formatted data
saps = pipeline(pathToJson)

# Initialize profit dicts and metadata
saps.initProfitDicts()

# Get profits for a certain length hold
saps.holdForTime(25)

# Cut missed rows from dataframe and append profits
saps.cutMissedProfits()
saps.appendProfitsToDf()

# Remove tickers from text
# saps.removeTickersFromText()
bank = saps.simulateProfits()



