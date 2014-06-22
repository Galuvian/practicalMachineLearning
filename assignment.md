Practical Machine Learing Assignment
========================================================

# Introduction

This assignment is to take a bunch of sensor data collected while subjects were performing barbell excercises, and predict the quality of the movements the subject made based on the test data.

# Data Preparation

## Data Source
The data training data is located here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The testing data is located here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The original source and a discussion on the data can be found here: http://groupware.les.inf.puc-rio.br/har

## Loading and cleaning

The data contains 160 columns.  Many of these are blank or NA for the vast majority of measurements.  Taking a look at the testing data, we see that only 60 columns are populated with actual data.  So we will use those columns for the learning algorithm and discard the rest.

Also, although the X column contains numbers, we won't use that because it seems that the training data is sorted on the 'classe' column and allowing the machine learning algorithm to use the X column could lead to wrong results.

Columns used:

```r

keepCols <- c("classe", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "cvtd_timestamp", "new_window", "num_window", "roll_belt", "pitch_belt", 
    "yaw_belt", "total_accel_belt", "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", 
    "accel_belt_x", "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", 
    "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", 
    "gyros_arm_x", "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y", 
    "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", 
    "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", "gyros_dumbbell_x", 
    "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y", 
    "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", 
    "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", "gyros_forearm_x", 
    "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", 
    "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")
```


The data is loaded as follows:

```r
training_file <- "pml-training.csv"
testing_file <- "pml-testing.csv"

full_training <- read.csv(training_file)

full_testing <- read.csv(testing_file)
```


## Cross Validation

Although there is some discussion on the class forum about the random forest algorithm not needing cross validation, for the purposes of this course it is important to understand the concept, so we will apply it here.   

A 10/90 split will be made with the data.  10% for training and the remaining 90% for a validation dataset.  After the validation dataset appears to be passing, we will apply the model to the 20 test cases for the assignment.

10% is a low number, but training a random forest is REALLY SLOW and 10% seems to produce results that are accurate enough.

The source data is not sequential, but does appear to be sorted by both the classe and user_name columns.  So we will use a partitioning technique that randomly mixes up the data.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(235458)


inTrain <- createDataPartition(y = full_training$X, p = 0.1, list = FALSE)

training <- full_training[inTrain, keepCols]
validation <- full_training[-inTrain, keepCols]
```


# Applying Machine Learning

## Approach

Reading the supporting documentation for the data, the original authors said that they used a Random Forest approach.  I will do the same here.

## Performance

We will enablemulti-threading to speed up the training.  


```r
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r

## I have 4 cores on my desktop, set this to whatever number you have
registerDoParallel(cores = 4)
```


## Training



```r
modFit <- train(classe ~ ., data = training, method = "rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r

```


## Validation

With a model trained, we can use the validation records to assess how accurate model is.


```r
valResults <- predict(modFit, newdata = validation)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
validation$answer <- valResults

```


## Out of Sample Error


```r
tmp <- validation[, c("user_name", "classe", "answer")]
tmp$match <- tmp$classe == tmp$answer

summary(tmp)
```

```
##     user_name    classe   answer     match        
##  adelmo  :3513   A:5019   A:5136   Mode :logical  
##  carlitos:2780   B:3413   B:3313   FALSE:369      
##  charles :3184   C:3100   C:3109   TRUE :17289    
##  eurico  :2790   D:2867   D:2883   NA's :0        
##  jeremy  :3042   E:3259   E:3217                  
##  pedro   :2349
```

```r

table(tmp$answer, tmp$classe)
```

```
##    
##        A    B    C    D    E
##   A 5011  125    0    0    0
##   B    8 3236   69    0    0
##   C    0   43 3023   43    0
##   D    0    9    8 2813   53
##   E    0    0    0   11 3206
```

```r

```





Of 17658 records in the validation set, 17289, or 97.9103% matched and 369 did not match.  The out of sample error rate is 2.0897%.

# Results of Test Set

Now that a high level of confidence has been shown in the model, we can apply it to the test data.


```r
answers <- predict(modFit, newdata = full_testing)
answers
```

```
##  [1] B A B A A E D A A A B C B A E E A B B B
## Levels: A B C D E
```


When these answers were submitted, they were all correct.  




