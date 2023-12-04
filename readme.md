# Information-theoretic Analysis of tracking data
Adapted by Martin Esparza

> ❗️ Disclaimer: This repo is tested on Linux. I am not sure how this will 
> work on windows. 

> ❗️ 2nd Disclaimer: Please take some time to understand the Poetry tool 
> (below) since there are custom packages needed to run these scripts

## Step 1: Activate ```poetry ``` environment
First of all, you should familiarize yourself with how the tool Poetry 
works. It is a great tool to manage package dependencies and make sure 
the project will run on your computer after running on mine!!!
Their [online documentation](https://python-poetry.org/docs/) is quite 
decent. Follow the instructions online to install poetry on your machine. 

Once you have poetry install in your machine you can test if its working by 
running in a terminal: 
```shell
$ poetry
```
and getting a response

Then, clone the [repository](https://github.com/martinesparza/TrackingInfoRate)
found in my github onto your local machine. You 
can use ``ssh`` or `https`, but I really recommend the first
```shell
$ git clone <paste-link-from-github>
```
Once this is done, you should see all the scripts from the repo on your 
machine

Now, being inside the folder of the repo, run:
```shell
$ poetry install
```
If there are no issues, this will install all the needed packages for the 
scripts to run. Its important to do this step well. 

Inside your repository folder again you can verify everything is correct 
this way. Please note the folders are from my own computer, just for 
example purposes
```shell
repos/TrackingInfoRate $ poetry version
# trackinginforate 0.1.0
```

> 👍 Success

Now you are ready to run the Info Theory pipeline on your data. We will go 
through it with an example. 

## Run the pipeline


I will give an explanation of what's going on but first, to run the code to 
extract feedback and feedforward measures:
```
$ poetry run python3 ITscript.py
```

This script takes as input data with columns as trials and rows as time 
points. There are approximately 420 time points which coming from sampling 
at 60 Hz. Then each column is a trial, and they are grouped into "cursor", 
"colour", and "target". In our analysis we don't actually make much use of 
the colour data, but we leave it there to make the scripts run well. 


