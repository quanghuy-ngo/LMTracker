# Anomaly-Detection-using-Autoencoders
An anomaly is a data point or a set of data points in our dataset that is different from the rest of the dataset. It may either be a too large value or a too small value. Anomalies describe many critical incidents like technical glitches, sudden changes, or plausible opportunities in the market. Anomalies are a very small fraction of the entire dataset. In this project, we look at how autoencoders can be used to detect anomalies. 

## Overview
This jupyter notebook explains how one can create an Autoencoder to detect Anomalies.

## Requirements
* Pyhton 3.x
* TensorFlow 2.x
* Pandas 
* Numpy
* Matplotlib

## Dataset used
The dataset used for this explanation is ECG5000 available [here](http://www.timeseriesclassification.com/Downloads/ECG5000.zip) link. The dataset contains ECG readings. This dataset contains the labels in the first column and the rest of the columns contain the features.

## Framing the Problem Statement
Here it is considered as a non time series problem where we have to detect anomalies in the dataset.

## Explanation
I have written an article explaining this at hello ML available [here](https://helloml.org/anomaly-detection-using-autoencoders/).

## How do I use this?
You can simply download the jupyter notebook and run it. Feel free to make changes and execute them to understand the concepts better.

## Contributions
You can make a contribution to it by making a pull request to it.
