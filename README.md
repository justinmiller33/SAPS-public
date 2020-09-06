# **Sentiment Analysis of Penny Stocks**

### **Data**
- dfXXXX.npy: Dataframe of scraped posts with identifier, content, and metadata.
- fDataInterXXXX.pkl: Dict of scraped interday financial data with identifiers.
- fDataIntraXXXX.pkl: Dict of scraped intraday financial data with identifiers.

### **Processes**

**extraction.py**
- Class to scrape data from multiple subreddits.
- Extracts post data and metadata from Reddit's PushShift API
- Uses Yahoo Finance API to extract relevant financial data
- Extracts matching dicts of intraday and interday profits

**simulation.py**
- Real time validation of trades by simulating policies
- Utilizes RobinHood API to get theoretical profits

### **Limitations**
- Trading Fees and Volume Limitations
- Dependency on volatile market behavior (COVID-19)

#### **Proof of Concept: Simulating profits from manual feature extraction.** 
![Proof Of Concept](https://github.com/justinmiller33/SAPS-public/blob/master/Proof%20Of%20Concept/pocWhole.png)
