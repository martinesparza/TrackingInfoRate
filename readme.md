### Information-theoretic Analysis of tracking data
This script takes tracking data as input and computes feedback, feedforward and total information from recorded behaviour, in accordance with Lam & Zénon, 2021. 

Input data are csv files "[filename].csv", with the samples in rows and the variables in columns. In its current form, the script takes mandatorily 30 trials and 3 variables: 1 output and 2 intputs. 

Output data are also csv files, names "output_[filename].csv", with one column per output variable and one row per trial. These files will be saved in an output/ directory within the working directory (needs to be created beforehand).

To run the code in terminal :
```
python ITscript.py -f "exampleData.csv"
```
or 
```
python ITscript.py -p "data/"
```
to provide a directory within which all files will be read and analysed. 

```
python ITscript.py -p "data/" -pl True
```
will also plot a figure of the results of each file. The figure window has to be closed for the code to continue running.

A. Zénon, August 2023