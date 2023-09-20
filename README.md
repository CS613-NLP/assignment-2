# Assignment-2

Repository for the Assignment 2 of course CS613, Natural Language Processing on Language Modeling and Smoothing.

## Table of Contents

- [Project Name](#Assignment-2)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [File Structure](#file-structure)
  - [Usage](#usage)

## About

In this Assignment, we had to train n-gram models from unigram to quadgram on the dataset provided. We had to implemenent different smoothing techniques and compare the preplexities of the models. Details of each task can be found here - [documentation](Tasks_and_Results.pdf)

## File Structure
The repository contains the following folders. Their descriptions are as follows:
- <b>average_preplexity</b>: average_preplexity: This folder contains csv files that contains all the average perplexities over all the models for both train and test dataset.
- <b>dataset</b>: This folder contains csv files that contain all the data. This includes the raw data, processed data, and train and test data subsets.
- <b>perplexities</b>: This folder contains subfolders for all the different smoothing techniques. Each subfolder (i.e., for each smoothing technique), contains csv files that contain the perplexities for each model after smoothing.
- <b>Plots</b>: This folder contains image files of plots for trends of different smoothing techniques over the different models.

The repository contains the following python files. Their descriptions are as follows:
- <b>NGramProcessor.py</b>: This file contains the class for the n-gram model. This includes all the methods to calculate the perplexities for the model as well.
- <b>ngram_train.py</b>: This file contains code that creates, trains, and gets the perplexities of the n-gram model.
- <b>plot_saver.py</b>: This file contains functions to plot and save the average perplexities.
- <b>preprocessing.py</b>: This file contains a function to preprocess the data and save it.

## Usage

Usage can be found in point 5 of the [documentation](Tasks_and_Results.pdf).
