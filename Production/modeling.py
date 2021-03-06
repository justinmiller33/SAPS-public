# Modeling.py
# Static File intended to research possible policies for use in simulatioin.py
# Does not modify data or simulate live policy enactment

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib import dates as md
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

2.3 clearBadProfits
() ; | insignificant |
Converting infs and nan data to 0% profit. Recording amount ignored
REQS: cutMissedProfits

5.0 simulateProfits
(sub, verbose) : | ~10 Seconds, ~ 20 seconds verbose |
Simulating a retrospective run over the last 6 months. Outputting ROI and trade logs.
REQS: clearBadProfits
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
        self.endToEnd = kwargs.get("endToEnd")
        
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

                    # If end to end trading:
                    if self.endToEnd:
                        
                        # Find Start unix
                        startUnix = dayListUnix[0]

                        # Get starting profit
                        startProfit = 1 + dataTemp[startUnix].iloc[-1]

                        # Assign new profit based off of starting profit
                        profit = (profit-startProfit) / startProfit + 1
                        
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
            self.data[sub]['raw']['postData'].index = saps.data[sub]['raw']['postData'].index.astype(str)

            # Make new df keeping only posts with profits
            self.data[sub]['raw']['postData'] = saps.data[sub]['raw']['postData'].loc[list(saps.profitDict[sub].keys())]

            
    # Adding profits to dataframe for easier manipulation
    def appendProfitsToDf(self):

        # Assert inter
        self.assertInter()

        # For each subreddit
        for sub in self.subList:

            # Create a new column with profit data
            saps.data[sub]['raw']['postData']['profit'] = list(saps.profitDict[sub].values())


    # Function to clear bad profits (nans and infs) 
    def clearBadProfits(self):
        infs = {}
        nans = {}

        for sub in saps.subList:
            
            infBool = saps.data[sub]['raw']['postData']['profit'] > 10000
            toDrop = list(infBool)
            saps.data[sub]['raw']['postData']['profit'][toDrop] = 1
            infs[sub] = sum(toDrop)
            
            nanBool = np.isnan(saps.data[sub]['raw']['postData']['profit'])
            toDrop = list(nanBool)
            saps.data[sub]['raw']['postData']['profit'][toDrop] = 1
            nans[sub] = sum(toDrop)

            # SHOULDNT GO HERE 
            # Just testing naive elimination
            # strLen = 50
            # self.data[sub]['raw']['postData'] = self.data[sub]['raw']['postData'].iloc[np.where(saps.data[sub]['raw']['postData']['text'].apply(lambda x: len(x) > strLen))[0]]
  
        return infs,nans


        
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
            


    """
    5.0
    Simulating Profits (retrospective)
    """
    def simulateProfits(self,sub,verbose):

        # Saving df
        df = self.data[sub]['raw']['postData']

        # Starting money and keys
        reserve = 1000
        reserveInitial = reserve
        bank = 0
        portfolio = 0
        tradeLog = {}
        holdingKeys = list()

        # Assigning Start Time
        # time1 = np.min(df.unix)-10
        # Start time at start of volatility 3/11/2020
        mbd = 1583884800
        nye = 1577836800
        nye19 = 1546300800
        time1 = df.unix[df.unix > nye19][-1] - 1
        time2 = time1 + 3600
        
        endTime = 1601000000
        print("started r/" + sub)
        print(self.data[sub]['md']['postData']['startDate'])
        while time2 < endTime:

            # Getting range of keys between those times (reversed to start with first buy
            keysInRange = list(df.unix[time1 < df.unix][time2 > df.unix].index)[::-1]
            # Adding those to holdingKeys
            holdingKeys = holdingKeys + keysInRange
            
            # Subtracting from bank ($1 EACH)
            investmentAmount = 1
            if bank >= investmentAmount * len(keysInRange):
                bank = bank - investmentAmount * len(keysInRange)

            else:
                reserve = reserve - (investmentAmount * len(keysInRange) - bank)
                bank = 0

             
            for key in keysInRange:
                tradeLog[key] = {}
                tradeLog[key]['timeOfPurchase'] = time2
                tradeLog[key]['boughtFor'] = investmentAmount
                
            
            # Starting loop to check for keys who's sell time has come
            timeToHold = (self.info['daysToHold'] + 2*self.info['daysToHold']/5) * 86400

            # Flag to indicate if all holds have been checked for this time
            checkedHold = False

            # While it hasn't checked
            while not checkedHold:

                # If we are are no longer holding stocks, end the run
                if not len(holdingKeys):

                    # Record how much we spend and out return on investments
                    #totalSpent = 0
                    #totalMade = 0
                    #for trade in tradeLog:
                     #   totalSpent = totalSpent + tradeLog[trade]['boughtFor']
                      #  totalMade = totalMade + tradeLog[trade]['soldFor']

                    # Display to user
                    print("Total Amount Invested: " + str(reserveInitial - reserve))
                    print("Total Return: " + str(bank))
                    print("Percent Return: " + str(100*(bank/(reserveInitial-reserve) - 1)))

                    # Set time to end to exit loop
                    time2 = endTime

                    # Return trade log for user to examine
                    return tradeLog

                    
                # If we are still holding stocks

                # Check the oldest key purchased
                keyToCheck = holdingKeys[0]

                # Get timestamp
                unixOfKey = df.unix[keyToCheck]

                # If the current time is greater than the time to hold
                if time2 > unixOfKey + timeToHold:

                    # Update the trade log with sold price and profit
                    tradeLog[keyToCheck]['soldFor'] = tradeLog[keyToCheck]['boughtFor'] * df.profit[keyToCheck] 
                    tradeLog[keyToCheck]['profit'] = df.profit[keyToCheck]

                    # Update our bank and total portfolio amount
                    bank = bank + tradeLog[keyToCheck]['soldFor']
                    
                   # portfolio = portfolio - tradeLog[keyToCheck]['boughtFor'] + tradeLog[keyToCheck]['soldFor']

                    # Update user of running bank amount
                    if verbose:
                        """
                        MATT: right now this is printing overall portfolio value
                        This is assuming we start with $10000 in cash, and reinvest
                        1/10000 of our portfolio back in. For the first few trades
                        this is $1, but when we start getting returns after 30 days
                        (or however many ur running it for) that changes. The 10000
                        is just arbitrary, not all of that gets invested and definitely
                        not all at once.

                        You can also change this print value to see how money in the
                        bank changes over time. Just change print(portfolio) to
                        print(bank)
                        """
                        print(bank)

                    # Get rid of that key (stop holding it)
                    del holdingKeys[0]

                    
                    continue

                # If we haven't reached the time to sell anyone continue
                else:
                    checkedHold = True

            # Update timestep
            time1 = time2
            time2 = time2 + 3600

        """
        6.0
        Visualizing
        """


"""
------
ACTION
------
"""

# Input local data path
pathToJson = "/home/justinmiller/Documents/OfflineDatasets/saps2.json"
# pathToJson = "/home/justinmiller/Documents/OfflineDatasets/saps.json"
# Initialize pipeline with json formatted data
saps = pipeline(pathToJson, endToEnd = True)

# Initialize profit dicts and metadata
saps.initProfitDicts()

# Get profits for a certain length hold
saps.holdForTime(30)

# Cut missed rows from dataframe and append profits
saps.cutMissedProfits()
saps.appendProfitsToDf()


# Clearing bad profits (nans, infs, etc)
infs, nans =  saps.clearBadProfits()

# Get tradeLogs for simulated trades
log = {}

"""
# Printing means
for sub in saps.subList:
	print(sub)
	print(np.nanmean(list(saps.profitDict[sub].values())))
"""
	
# For each log
saps.subList = ["Investing","Stocks"]
for sub in saps.subList:
    
    # Simulate profits
    tradeLog = saps.simulateProfits(sub, verbose = False)
    
    # Save tradeLog in master dict
    log[sub] = tradeLog

# TO visualize performance over time (outside of class)
def visualizePerf(subList,log):
    # Datetime formatting
    import datetime as dt

    # For each sublist 
    for sub in subList:
        date = np.array([])
        ps = np.array([])

        # For each key append parallel lists with date and profits
        for key in list(log[sub].keys()):
            ps = np.append(ps, log[sub][key]['profit'])
            date = np.append(date, dt.datetime.fromtimestamp(log[sub][key]['timeOfPurchase']))

        # Format plot to plot for datetimes
        ax = plt.gca()
        xfmt = md.DateFormatter('%m-%d')
        ax.xaxis.set_major_formatter(xfmt)

        # Plot cumulative (y value is a proxy for profit)
        plt.plot(date,np.cumsum(ps-1))
        plt.title(sub)
        plt.show()

