# Code structure

The code is organised around 4 files:

**rf.labels.py**: Definition of labels used in the Random Forest model. -1 for a downward movement, +1 for an uppward movement and 0 if it's stationary.The labels are calculated over 3 times horizons: 5 seconds ahead, 10 seconds ahead and 20 seconds ahead

**rf.features.py**: Definition of features used in the Random Forest model

**rf.calibration.py**: Calibration and test of the Random Forest model

**useful.py**: Various usel functions (work in progress)
