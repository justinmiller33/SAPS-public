# SAPS Json handler
# Converting npy and pkl files to a single json heirarchy for upload

# Modules
import time
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime
import requests
import pandas as pd
import re
import pickle
import os
import json

# Function to load a dataframe with info on a subreddit
def loadDf(pathToFile):
    df = np.load(pathToFile, allow_pickle = True)
    df = pd.DataFrame(df, columns = ['ticker','title','text','type','datetime'])

    return df

# Function to read pickle file
def readPickle(pathToFile):
    file = open(pathToFile, "rb")
    fData = pickle.load(file)
    file.close()

    return fData

# Function to load existing json
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

# Convert a list of times to unix
def timeToUnix(timeList):
    toUnix = lambda x : int(x.timestamp())
    times = list(map(toUnix, timeList))

    return times
    
# Function to deserialize fData (remove pandas objects and convert timestamps to unix)
def deserialize(fData):

    # Looping through keys
    for key in list(fData.keys()):

        # Initialize numpy matrix
        postData = np.zeros((len(fData[key]),2))

        # First Column: Time Data
        # Converting to Unix
        timeList = timeToUnix(list(fData[key].index))
        postData[:,0] = timeList
        
        # Second Column: Profit Data
        postData[:,1] = fData[key].values

        # Writing postData over old pandas series
        fData[key] = postData.tolist()

    return fData

# Writing post data  metadata to jsonDict
def writeDfMd(jsonDict, dfName, df):

    # Including subreddit, number of posts, and start+end date of scrape
    jsonDict[dfName]['md']['postData']['subreddit'] = "r/" + dfName
    jsonDict[dfName]['md']['postData']['posts'] = len(df)
    jsonDict[dfName]['md']['postData']['startDate'] = dt.fromtimestamp(int(df[-1,-1])).strftime('%Y-%m-%d %H:%M:%S')
    jsonDict[dfName]['md']['postData']['endDate'] = dt.fromtimestamp(int(df[0,-1])).strftime('%Y-%m-%d %H:%M:%S')

    return jsonDict

# Writing financial data  metadata to jsonDict
def writeFDataMd(jsonDict, dfName, finType, fData):

    # Including subreddit, number of posts, and start+end date of scrape
    jsonDict[dfName]['md'][finType]['postsWithFinData'] = len(fData)
    jsonDict[dfName]['md'][finType]['keys'] = list(fData.keys())
    
    return jsonDict
    
""" ACTION """

# Loading recent json data
pathToJson = "/home/justinmiller/Documents/OfflineDatasets/sapsRecent.json"
recentData = loadJson(pathToJson)


# Initializing dict as old data that will turn into json file
jsonDict = {}

# Getting all files in data directory
dataDir = '/home/justinmiller/devel/SAPS-public/Data/'
files = np.array(os.listdir(dataDir))

# Getting lists of pkl and numpy files
ispkl = lambda x : x[-4:] == '.pkl'
isnpy = lambda x : x[-4:] == '.npy'

pklList = files[list(map(ispkl, files))]
npyList = files[list(map(isnpy, files))]

# Looping through each post data file
for npy in npyList:

    # Extracting name and data
    dfName = npy[2:-4]
    df = loadDf(dataDir + npy)

    # Reverting df to numpy (Redundant... but guarentees consistency)
    df = df.to_numpy()

    # Checking to see if this data is the same as the old json
    # If height doesn't line up... its not the same df
    try:
        sameDf = (np.array(df) == np.array(recentData[dfName]['raw']['postData']))[:,0:4].all()
    except:
        sameDf = False
        
    print(sameDf)
    # Initializing empty dict for that channel
    jsonDict[dfName] = {}

    # Initializing md and raw
    jsonDict[dfName]['md'] = {}
    jsonDict[dfName]['raw'] = {}

    # Initializing subgroups for md and raw
    jsonDict[dfName]['md']['postData'] = {}
    jsonDict[dfName]['md']['inter'] = {}
    jsonDict[dfName]['md']['intra'] = {}
    jsonDict[dfName]['raw']['postData'] = {}
    jsonDict[dfName]['raw']['inter'] = {}
    jsonDict[dfName]['raw']['intra'] = {}
    
    # Convert post times to unix
    timeList = df[:,-1]
    timeList = timeToUnix(timeList)
    df[:,-1] = timeList
    
    # If df is the same as the old data
    if sameDf:

        # Swap in recent data, md and raw will stay the same
        jsonDict[dfName] = recentData[dfName]
        print(dfName + " stayed the same")
        continue

    # If not
    else:
        # Old data df list and new df list
        recentList = np.array(recentData[dfName]['raw']['postData']).tolist()
        dfList = df.tolist()

        # Full list of df
        dfList.extend(recentList)

        # Adding to dict
        jsonDict[dfName]['raw']['postData'] = dfList

        # Switching back to numpy array
        df = np.array(dfList)
        
        # Write df metadata
        jsonDict = writeDfMd(jsonDict, dfName, df)
        print(dfName + " changed")
        

    
# Looping through each finacial data file
for pkl in pklList:

    
    # Extracting name and data
    dfName = pkl[10:-4]
    
    # Finding its type (inter, intra)
    finType = pkl[5:10].lower()

  
    if sameDf:

        jsonDict[dfName]['raw'][finType] = recentData[dfName]['raw'][finType]
        print(dfName + " fin stayed the same")
        
    else:
        # Reading data
        fData = readPickle(dataDir + pkl)

        # Deserialize fData
        fData = deserialize(fData)

        # Write the raw data
        jsonDict[dfName]['raw'][finType] = fData

        # Write financial metadata
        jsonDict = writeFDataMd(jsonDict, dfName, finType, fData) 
        print(dfName + " fin changed")
        
print("Finished Creating Dict, writing to json... ")

# Writing to json file
"""
with open(dataDir + 'sapsAuto.json', 'w') as file:
    json.dump(jsonDict, file)
    """
