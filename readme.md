### Information-theoretic Analysis of tracking data
This script takes tracking data as input and computes feedback, feedforward and total information from recorded behaviour, in accordance with Lam & Zénon, 2021. 

Input data are csv files "[filename].csv", with the samples in rows and the variables in columns. In its current form, the script takes mandatorily 30 trials and 3 variables: 1 output and 2 intputs. 
Output data are also csv files, names "output_[filename].csv", with one column per output variable and one row per trial. 

To run the code in terminal :
```
python ITscript.py -f "vassiliadisData.csv"
```
A. Zénon, August 2023