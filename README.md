# lanl-auth-cybersecurity

This is a data analysis of  a cybersecurity dataset from Los Alamos National Laboratory. The dataset and description of the dataset can be found at http://csr.lanl.gov/data/cyber1/ . 

Here I only worked with the file auth.txt.gz that represents authentication logs captured for the dataset. This file contains 8 columns of data and my goal was to use first seven columns to predict the values in the last column.

To accomplish this task, I used 64-bit Linux computer with 8GB of RAM. 

Below I describe the steps I took for this analysis.

## First look

For speed and efficiency, I worked with auth.txt.gz directly. First I created a small subsample of the file just to get the feel for the data.
I created such file using bash commands:
```
zcat auth.txt.gz | cat -n | grep '00000    ' > sample2.csv
cat sample2.csv | cut -f 2 > sample2v.csv
```
After that I used jupyter notebook for my initial analysis that one can see in 
```
exploration.ipynb 
```
also exported to `exploration.pdf` in case you do not have ipython installed.	

## Trying machine learning

I collected roughly 400,000 data points randomly sampled from auth.txt.gz so that the number of "fails" and "success" were about the same. The code for this task can be found in 
```
multisample.py
```

I then used jupyter notebook to create 55 features and tried a few classifiers to see how well I can predict success for authentication. This code and results can be found in

```
machine learning.ipynb 
```
also exported to `machine learning.pdf` in case you do not have ipython installed.


## Analysis

So far analysis has been done on very small subset of auth.txt.gz. I now wanted to test my result on bigger subset. I used 
```
multisample.py
```
to generate multiple non-overlapping subsets of auth.txt.gz and tested my model. The results can be found in

```
further analysis.ipynb 
```
also exported to `further analysis.pdf` in case you do not have ipython installed.
