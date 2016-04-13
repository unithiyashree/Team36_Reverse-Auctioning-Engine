# Team36_Reverse-Auctioning-Engine
IRE Major Project
Step 1:
On executing "createLabels.py",we create a new file bids_labelled.csv in which each bid is assigned one of the following classes:
0-Human
1-Robot
2-Unknown

Step 2:
We then extract the following features from it by running "extract_features.py"
  - total number of bids placed (log)
  - total number of auctions of participation (log)
  - total number of countries from which bids were placed (log)
  - total number of ips from which bids were placed (log)
  - total number of urls from which bids were placed (log)
  - total number of device types from which bids were placed (log)
  - total number of "wins" (last bid placed in auction) (log)
  - win percent
  - mean bids per auction
  - bidding stage (whether bids were placed earlier or later in auction)
  - mean time between own bids
  - mean time between own bid and the previous bid placed by a competitor
  - mean number of competitors per auction of participation
  - mean number of bots per auction of participation (not including self)
This step results in the creation of two files "test_features.csv" and "train_features.csv"

Step 3:
We then apply the following three classifiers on training set to build the model and predict results :
  -Logistic Regression
  -RandomForestClassifier
  -BaggingClassifier
The resulting ensemble gave accuracy of 93.064%

