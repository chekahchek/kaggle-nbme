# NBME - Score Clinical Patient Notes

## Introduction

This repository contains the inference code used in the [NBME - Score Clinical Patient Notes](https://www.kaggle.com/c/nbme-score-clinical-patient-notes) kaggle competition. This solution is ranked 120th out of 1471 teams (top 9%). More explanation on the code can be found [here](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/322853).

## Setting up  

Before running the code, you would need to download the dataset as well as the models used in the competition. 
```bash

$ export KAGGLE_USERNAME=[your kaggle username]
$ export KAGGLE_KEY=[your kaggle api key]
$ chmod +x ./setup.sh
$ ./setup.sh
```

## Run inference

```bash
$ python src/main.py
```
