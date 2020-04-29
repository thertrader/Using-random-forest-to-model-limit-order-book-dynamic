# Using random forest to model limit order book dynamic
I use the random forest algorithm to forecast mid price dynamic over short time horizon i.e. a few seconds ahead. This is of particular interest to market makers to skew their bid/ask spread in the direction of the most favorable outcome. Most if not all the literature on the topic (see references below) focuses on applying straight out of the box algorithm to create forecast at any point in time. The problem in a real life environment is different. A market maker can provide a standard bid/ask spread most of the time and only when she/he has a statistical hedge she/he can skew the spread in the direction given by the model. This is what I try to do here: creating a forecast only when a statistical hedge exists

I used Python scikit-learn  library. This GitHub repo contains the code, some sample data and the associated explanations (code comments). I'm happy for anyone to re-use my work as long as proper reference to it is made.

I wanted to thank LOBSTER for providing the dataset used here

## Code structure
The code is organised around 4 files:

**rf.labels.py**: This file defines the labels used in the Random Forest model. -1 for a downward movement, +1 for an uppward movement and 0 if it's stationary.The labels are calculated over 3 times horizons: 5 seconds ahead, 10 seconds ahead and 20 seconds ahead

**rf.features.py**: Definition of features used in the Random Forest modeling 
FEATURES DEFINITION:
- 0    - BASIC PARAMETERS
- 1.1  - START & STOP TIMESTAMPS: This is ran once for all for each file. Look for the timestamp 5sec backward. This is used for deciles definition.
- 1.2  - IMBALANCE: Absolute levels & deciles ranking compared to last 5min
- 1.3  - MID PRICE, B/A PRICE SPREAD, B/A VOLUME SPREAD: Absolute levels & deciles ranking compared to last 5min
- 1.4  - PRICE DIFFERENCES ACCROSS LEVELS: Bid & Ask differences between order book levels (1 to 10). Absolute levels & deciles ranking compared to last 5 min
- 1.5  - PRICE AVERAGE & VOLUME AVERAGE: Mean calculated across order book levels. Deciles ranking compared to last 5 min
- 1.6  - ACCUMULATED DIFFERENCES PRICE & VOLUME: Sum of all bid minus sum of all asks across levels. Deciles ranking compared to last 5 min
- 1.7  - BID, ASK, BID SIZE & ASK SIZE DERIVATIVEs: Last compared to price 1 sec. ago. Deciles ranking compared to last 5 min 
- 1.8  - AVERAGE TRADE INTENSITY: Count events of type 1,2,3 (see definition above) over the last second and deciles ranking compared to last 5 min
- 1.9  - RELATIVE TRADE INTENSITY 10s: Count events of type 1,2,3 over the last 10 sec. ** INTERMEDIARY STEP **
- 1.10 - RELATIVE TRADE INTENSITY 900s: Count events of type 1,2,3 over the last 900 sec. ** INTERMEDIARY STEP **
- 1.11 - RELATIVE TRADE INTENSITY: Relative count events 10 sec./900 sec. Deciles ranking compared to last 5 min

EVENT TYPE (see LOBSTER website - [https://lobsterdata.com/](https://lobsterdata.com/)):
1. New limit order 
2. Partial deletion of limit order
3  Total cancellation of limit order

