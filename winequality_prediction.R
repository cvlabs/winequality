#############################################################
# The data source for this R script is the Wine Quality data 
# from the UCI Machine learning repository. The observations
# for red wines are in a separate data file from observatons
# for white wines. Separate models were created for the two
# types of wines.
#
# The script predicts the quality for each observation in 
# the validation set using:
# 1. Support Vector Machine
# 2. Principal Component Analysis + Support Vector Machine
# 3. Random Forests
#
# Author: Claudia Valdeavella
# Date: 7 Feb 2019
#############################################################

# load the required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# svm is in the e1071 package
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")

# define ACCURACY() function 
ACCURACY <- function(true_ratings, predicted_ratings) {
  mean(true_ratings == predicted_ratings)
}

# read data files
red <- read.csv('winequality-red.csv')
white <- read.csv('winequality-white.csv')

# split into training and test sets
# separate into training (0.9) and test (0.1) sets
set.seed(1)
test_r_index <- createDataPartition(y = red$quality, times = 1, p = 0.1, list = FALSE)
red_train <- red[-test_r_index,]
red_test <- red[test_r_index,]
# scale the data before modeling 
X_train_red <- scale(red_train[, 1:11])
y_train_red <- red_train[, 12]
X_test_red <- scale(red_test[, 1:11])
y_test_red <- red_test[, 12]
# repeat the procedure for the white wine data
set.seed(1)
test_w_index <- createDataPartition(y = white$quality, times = 1, p = 0.1, list = FALSE)
white_train <- white[-test_w_index,]
white_test <- white[test_w_index,]

X_train_white <- scale(white_train[, 1:11])
y_train_white <- white_train[, 12]
X_test_white <- scale(white_test[, 1:11])
y_test_white <- white_test[, 12]

#### Principal Component Analysis ####
# run PCA on the red wine dataset
# allow the prcomp() function to scale the data
pca_red <- prcomp(red_train[, 1:11], scale=TRUE)
# calculate the proportion of variance explained by each principal component
var_explained_red <- cumsum(pca_red$sdev^2/sum(pca_red$sdev^2))
# plot the proportion of variance explained
plot(var_explained_red, xlab="Principal Component", ylab="Cumulative Sum of Prop Variance Explained", type="b", main="Red Wine PCA")
# print out the information in a table
summary(pca_red)

# run PCA on the white wine dataset 
pca_white <- prcomp(white_train[, 1:11], scale=TRUE)
# calculate the proportion of variance explained by each principal component
var_explained_white <- cumsum(pca_white$sdev^2/sum(pca_white$sdev^2))
# plot the proportion of variance explained
plot(var_explained_white, xlab="Principal Component", ylab="Cumulative Sum of Prop Variance Explained", type="b", main="White Wine PCA")
# print out the information in a table
summary(pca_white)

#### Support Vector Machines ####

# SVM analysis of the red wine dataset
# tune the cost and gamma parameters for nonlinear svm
set.seed(100)
svm_tune_nonlin_red <- tune(svm, train.x=X_train_red, train.y=y_train_red, 
                            kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

bestCost <- svm_tune_nonlin_red$best.parameters$cost
bestGamma <- svm_tune_nonlin_red$best.parameters$gamma
# build the model, have svm scale the data prior to building the model
svmfit_nonlin_red <- svm(red_train[,1:11], as.factor(y_train_red), kernel= "radial",  cost=bestCost, gamma=bestGamma, scale=TRUE, probabilities="TRUE")

# predict the quality of the wine samples in the test set
svm_nonlin_pred_red <- predict(svmfit_nonlin_red, red_test[,1:11], scale=TRUE, probabilities = TRUE)

# create the confusion matrix
# use the table() function to create the confusion matrix because some of the levels are not populated in test set

table(svm_nonlin_pred_red, y_test_red)

# calculate the accuracy
svm_11var_acc_red <- ACCURACY(y_test_red, svm_nonlin_pred_red)

# SVM analysis of the white wine dataset
# tune the cost and gamma parameters for nonlinear svm
set.seed(100)
svm_tune_nonlin_white <- tune(svm, train.x=X_train_white, train.y=y_train_white, 
                              kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

bestCost <- svm_tune_nonlin_white$best.parameters$cost
bestGamma <- svm_tune_nonlin_white$best.parameters$gamma
# build the model, have svm scale the data prior to building the model
svmfit_nonlin_white <- svm(white_train[,1:11], as.factor(y_train_white), kernel= "radial",  cost=bestCost, gamma=bestGamma, scale=TRUE, probabilities="TRUE")

# predict the quality of the wine samples in the test set
svm_nonlin_pred_white <- predict(svmfit_nonlin_white, white_test[,1:11], scale=TRUE, probabilities = TRUE)

# create the confusion matrix
table(svm_nonlin_pred_white, y_test_white)

# calculate the accuracy
svm_11var_acc_white <-ACCURACY(y_test_white, svm_nonlin_pred_white)

#### PCA (PC1-PC8) + SVM ####

# Red Wine projection of observations in the training and test sets onto the principal component vectors (PC1-PC8)
# Train set from PCA
red_train_pca_8 <- as.matrix(X_train_red[, 1:11]) %*% pca_red$rotation[, 1:8]
red_train_pca_8 <- as.data.frame(red_train_pca_8)
colnames(red_train_pca_8) <- paste("PC_", 1:8, sep="")

# Test set from PCA
red_test_pca_8 <- as.matrix(X_test_red[, 1:11]) %*% pca_red$rotation[, 1:8]
red_test_pca_8 <- as.data.frame(red_test_pca_8)
colnames(red_test_pca_8) <- paste("PC_", 1:8, sep="")    

# Red Wine SVM analysis with 8 principal component vectors
# tune the cost and gamma parameters
set.seed(100)
svm_tune_nonlin_red_8 <- tune(svm, train.x=red_train_pca_8, train.y=y_train_red, 
                            kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

bestCost <- svm_tune_nonlin_red_8$best.parameters$cost
bestGamma <- svm_tune_nonlin_red_8$best.parameters$gamma
# build the model using the best parameter estimates
svmfit_nonlin_red_8 <- svm(red_train_pca_8, as.factor(y_train_red), kernel= "radial",  cost=bestCost, gamma=bestGamma, scale=TRUE, probabilities="TRUE")

# predict the wine quality
svm_nonlin_pred_red_8 <- predict(svmfit_nonlin_red_8, red_test_pca_8, scale=TRUE, probabilities = TRUE)

# create the confusion matrix
table(svm_nonlin_pred_red_8, y_test_red)

# calculate the accuracy
pca8_svm_red_acc <- ACCURACY(y_test_red, svm_nonlin_pred_red_8)

# White Wine projection of observations in the training and test sets onto the principal component vectors (PC1-PC8)
# Train set from PCA
white_train_pca_8 <- as.matrix(X_train_white[, 1:11]) %*% pca_white$rotation[, 1:8]
white_train_pca_8 <- as.data.frame(white_train_pca_8)
colnames(white_train_pca_8) <- paste("PC_", 1:8, sep="")

# Test set from PCA
white_test_pca_8 <- as.matrix(X_test_white[, 1:11]) %*% pca_white$rotation[, 1:8]
white_test_pca_8 <- as.data.frame(white_test_pca_8)
colnames(white_test_pca_8) <- paste("PC_", 1:8, sep="") 

# White Wine SVM analysis with 8 principal component vectors
# tune the cost and gamma parameters
set.seed(100)
svm_tune_nonlin_white_8 <- tune(svm, train.x=white_train_pca_8, train.y=y_train_white, 
                              kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

bestCost <- svm_tune_nonlin_white_8$best.parameters$cost
bestGamma <- svm_tune_nonlin_white_8$best.parameters$gamma
# build the model using the best estimates for cost and gamma
svmfit_nonlin_white_8 <- svm(white_train_pca_8, as.factor(y_train_white), kernel= "radial",  cost=bestCost, gamma=bestGamma, scale=TRUE, probabilities="TRUE")

# make predictions for the test data
svm_nonlin_pred_white_8 <- predict(svmfit_nonlin_white_8, white_test_pca_8, scale=TRUE, probabilities = TRUE)

# create the confusion matrix
table(svm_nonlin_pred_white_8, y_test_white)

# calculate the accuracy
pca8_svm_white_acc <- ACCURACY(y_test_white, svm_nonlin_pred_white_8)

# Red Wine projection of observations in the training and test sets onto the principal component vectors (PC1-PC10)
# Train set from PCA
red_train_pca_10 <- as.matrix(X_train_red[, 1:11]) %*% pca_red$rotation[, 1:10]
red_train_pca_10 <- as.data.frame(red_train_pca_10)
colnames(red_train_pca_10) <- paste("PC_", 1:10, sep="")

# Test set from PCA
red_test_pca_10 <- as.matrix(X_test_red[, 1:11]) %*% pca_red$rotation[, 1:10]
red_test_pca_10 <- as.data.frame(red_test_pca_10)
colnames(red_test_pca_10) <- paste("PC_", 1:10, sep="")    

#### PCA (PC1-PC10) + SVM ####

# Red Wine SVM analysis with 10 principal component vectors
# tune the cost and gamma parameters
set.seed(100)
svm_tune_nonlin_red_10 <- tune(svm, train.x=red_train_pca_10, train.y=y_train_red, 
                               kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

bestCost <- svm_tune_nonlin_red_10$best.parameters$cost
bestGamma <- svm_tune_nonlin_red_10$best.parameters$gamma
# build the model using the best parameter estimates
svmfit_nonlin_red_10 <- svm(red_train_pca_10, as.factor(y_train_red), kernel= "radial",  cost=bestCost, gamma=bestGamma, scale=TRUE, probabilities="TRUE")

# predict the quality for the test data
svm_nonlin_pred_red_10 <- predict(svmfit_nonlin_red_10, red_test_pca_10, scale=TRUE, probabilities = TRUE)

# create the confusion matrix
table(svm_nonlin_pred_red_10, y_test_red)

# calculate the accuracy
pca10_svm_red_acc <- ACCURACY(y_test_red, svm_nonlin_pred_red_10)

# White Wine projection of observations in the training and test sets onto the principal component vectors (PC1-PC10)
# Train set from PCA
white_train_pca_10 <- as.matrix(X_train_white[, 1:11]) %*% pca_white$rotation[, 1:10]
white_train_pca_10 <- as.data.frame(white_train_pca_10)
colnames(white_train_pca_10) <- paste("PC_", 1:10, sep="")

# Test set from PCA
white_test_pca_10 <- as.matrix(X_test_white[, 1:11]) %*% pca_white$rotation[, 1:10]
white_test_pca_10 <- as.data.frame(white_test_pca_10)
colnames(white_test_pca_10) <- paste("PC_", 1:10, sep="") 

# White Wine SVM analysis with 10 principal component vectors
# tune the cost and gamma parameters
set.seed(100)
svm_tune_nonlin_white_10 <- tune(svm, train.x=white_train_pca_10, train.y=y_train_white, 
                                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

bestCost <- svm_tune_nonlin_white_10$best.parameters$cost
bestGamma <- svm_tune_nonlin_white_10$best.parameters$gamma
# build the model using the best parameter estimates
svmfit_nonlin_white_10 <- svm(white_train_pca_10, as.factor(y_train_white), kernel= "radial",  cost=bestCost, gamma=bestGamma, scale=TRUE, probabilities="TRUE")

# predict the quality of the test set 
svm_nonlin_pred_white_10 <- predict(svmfit_nonlin_white_10, white_test_pca_10, scale=TRUE, probabilities = TRUE)

# create the confusion matrix
table(svm_nonlin_pred_white_10, y_test_white)

# calculate the accuracy
pca10_svm_white_acc <- ACCURACY(y_test_white, svm_nonlin_pred_white_10)

#### Random Forests ####

# Red Wine analysis using Random Forests
# tune parameters
set.seed(100)
red_rf <- train(red_train[,1:11], as.factor(y_train_red), method="Rborist", nTree=700, preProcess = c("scale"), tuneGrid=data.frame(predFixed=seq(2,11,1), minNode=seq(5,250,25)))

# make predictions for the test data
red_rf_pred_raw <- predict(red_rf, red_test[,1:11], as.factor(y_test_red), type="raw")

# calculate accuracy
rf_red_acc<- ACCURACY(y_test_red, red_rf_pred_raw)

# White Wine analysis using Random Forests
# tune parameters
set.seed(100)
white_rf <- train(white_train[,1:11], as.factor(y_train_white), method="Rborist", nTree=700, preProcess = c("scale"), tuneGrid=data.frame(predFixed=seq(2,11,1), minNode=seq(5,250,25)))

# make predictions for the test set
white_rf_pred_raw <- predict(white_rf, white_test[,1:11], as.factor(y_test_white), type="raw")

# calculate accuracy
rf_white_acc<- ACCURACY(y_test_white, white_rf_pred_raw)

# Compare the different models using accuracy as the metric
# round accuracy values to 4 digits
svm_11var_acc_red <- round(svm_11var_acc_red, digits=4)
svm_11var_acc_white <- round(svm_11var_acc_white, digits=4)
pca8_svm_red_acc <- round(pca8_svm_red_acc, digits=4)
pca8_svm_white_acc <- round(pca8_svm_white_acc, digits=4)
pca10_svm_red_acc <- round(pca10_svm_red_acc, digits=4)
pca10_svm_white_acc <- round(pca10_svm_white_acc, digits=4)
rf_red_acc <- round(rf_red_acc, digits=4)
rf_white_acc <- round(rf_white_acc, digits=4)

# build a table
accuracy_results <- data_frame(method = "svm 11 var", red = svm_11var_acc_red, white = svm_11var_acc_white)
accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="pca (1-8) + SVM",  
                                         red = pca8_svm_red_acc, white = pca8_svm_white_acc))
accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="pca (1-10) + SVM",  
                                         red = pca10_svm_red_acc, white = pca10_svm_white_acc))
accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="random forests",  
                                         red = rf_red_acc, white = rf_white_acc))
accuracy_results %>% knitr::kable()
