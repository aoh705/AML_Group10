library(readr)
library(caret)
library(tidymodels)
library(tidyverse)
library(randomForest)
library(xgboost)
library(keras)
library(vip)
library(pROC)
library(dplyr)
library(ggplot2)

### PURPOSELY MESSING UP THE DATASET
chess_games_data <- read.csv("Documents/Chess Win Prediction/club_games_data.csv")

set.seed(1)
# No. of rows to set as NA (0.0001% of your data)
num_rows_to_na <- ceiling(nrow(chess_games_data) * 0.0001)
# Columns to set as NA
columns_to_na <- c("white_rating", "black_rating", "rated", "time_class")

# Introduce NA values randomly across specified columns
for (col in columns_to_na) {
  # Randomly select rows for each column
  rows_to_na <- sample(1:nrow(chess_games_data), num_rows_to_na, replace = FALSE)
  # Set selected rows to NA for the column
  chess_games_data[rows_to_na, col] <- NA
}

write.csv(chess_games_data, "chess_games_data.csv", row.names = FALSE)






### DATA EXPLORATION

# See the first 6 rows
head(chess_games_data)
## The count of missing values in the dataset
sum(is.na(chess_games_data))
# Find the structure
str(chess_games_data)
# Print unique values in the white_result column
unique(chess_games_data$white_result)
# Summary
summary(chess_games_data)

### DATA PREPROCESSING

# Dropping the "black_result" column and setting the "white_result" column to
# either 1 or 0.
chess_games_data <- chess_games_data[, !(names(chess_games_data) %in% c("black_result"))]

chess_games_data <- chess_games_data %>%
  mutate(white_result = case_when(
    white_result %in% c("win", "threecheck", "kingofthehill") ~ "win",
    white_result %in% c("checkmated", "timeout", "resigned", "abandoned") ~ "loss",
    TRUE ~ "draw" # Assumes all other cases are draws
  ))

# Data Splitting

set.seed(1)      

index <- createDataPartition(chess_games_data$`white_result`, p = 0.70, list = FALSE)
games_train <- chess_games_data[index, ]
games_test <- chess_games_data[-index, ]

# Combine the datasets into one data frame with an additional column indicating the dataset source
combined_df <- rbind(data.frame(data = 'white_result', dataset = 'Full', value = chess_games_data$`white_result`),
                     data.frame(data = 'white_result', dataset = 'Train', value = games_train$`white_result`),
                     data.frame(data = 'white_result', dataset = 'Test', value = games_test$`white_result`))
combined_df <- combined_df %>%
  group_by(dataset, value) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count)) %>%
  ungroup()

ggplot(combined_df, aes(x = value, y = percentage, fill = dataset)) +
  geom_bar(position = "dodge", stat = "identity") +
  theme_minimal() +
  labs(title = "Distribution of Target Feature 'white_result' across Datasets",
       x = "Target Feature 'white_result' Value",
       y = "Percentage") +
  scale_fill_manual(values = c("Full" = "purple", "Train" = "blue", "Test" = "green"))

# Preprocessing with Recipes
blueprint <- recipe(white_result ~ ., data = games_train) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_log(all_numeric_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  step_other(all_nominal(), threshold = 0.01, other = "other") %>%
  step_dummy(all_nominal_predictors())

blueprint_prep <- prep(blueprint, training = games_train)

transformed_train <- bake(blueprint_prep, new_data = games_train)

transformed_test <- bake(blueprint_prep, new_data = games_test) ######## FOR SOME REASON, THIS TAKES FOREVER TO RUN



### RANDOM FOREST

resample_1 <- trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)    # adds more metrics to the model

hyper_grid <- expand.grid(mtry = c(13, 15, 17),
                          splitrule = c("gini", "extratrees"),
                          min.node.size = c(5, 7, 9))

rf_fit <- train(white_result ~ .,
                data = transformed_train, 
                method = "ranger",
                verbose = FALSE,
                trControl = resample_1, 
                tuneGrid = hyper_grid,
                metric = "ROC")


ggplot(rf_fit, metric = "Sens")   # Sensitivity
ggplot(rf_fit, metric = "Spec")   # Specificity
ggplot(rf_fit, metric = "ROC")    # AUC ROC


fitControl_final <- trainControl(method = "none",
                                 classProbs = TRUE)

RF_final <- train(white_result ~., 
                  data = transformed_train,
                  method = "ranger",
                  trControl = fitControl_final,
                  metric = "ROC",
                  tuneGrid = data.frame(.mtry = 13,
                                        .min.node.size = 9,
                                        .splitrule = "gini"))


# Training set results

RF_pred_train <- predict(RF_final, newdata = transformed_train)

RF_train_results <- confusionMatrix(transformed_train$white_result, RF_pred_train)

print(RF_train_results)

# Test set results

RF_pred_test <- predict(RF_final, newdata = transformed_test)

RF_test_results <- confusionMatrix(transformed_test$white_result, RF_pred_test)

print(RF_test_results)


### XG Boost

dtrain <- xgb.DMatrix(data = as.matrix(transformed_train[, -which(names(transformed_train) == "white_result")]),
                      label = as.numeric(transformed_train$white_result) - 1)

dtest <- xgb.DMatrix(data = as.matrix(transformed_test[, -which(names(transformed_test) == "white_result")]),
                     label = as.numeric(transformed_test$white_result) - 1)

hyper_grid <- expand.grid(
  eta = c(0.05, 0.1),
  max_depth = c(3, 6),
  min_child_weight = c(1, 2),
  subsample = c(0.5, 0.75),
  colsample_bytree = c(0.5, 0.75),
  gamma = c(0, 0.1),
  stringsAsFactors = FALSE  # To keep the grid as a dataframe of strings
)


# Initialize an empty dataframe to store results
results <- data.frame()

# Loop over each set of parameters in the grid
for(i in 1:nrow(hyper_grid)) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i],
    gamma = hyper_grid$gamma[i]
  )
  
  # Perform cross-validation
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0  # Change to 1 for more detailed output
  )
  
  # Record the best score and corresponding parameters
  best_score <- max(cv$evaluation_log$test_auc_mean)
  results <- rbind(results, cbind(hyper_grid[i, ], score = best_score))
}

# Review the results
results <- results[order(-results$score), ]  # Sort by score descending
print(results)

best_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 2,
  subsample = 0.50,
  colsample_bytree = 0.75,
  gamma = 0
)

# Train final model on the entire training dataset
final_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = 100  # Or use the best iteration number from your CV results
)

# Predictions on the training data
train_preds_prob <- predict(final_model, dtrain)
train_preds <- ifelse(train_preds_prob > 0.5, 1, 0)  # Convert probabilities to binary class output

# Prepare the testing data as DMatrix
dtest <- xgb.DMatrix(data = as.matrix(transformed_test[, -which(names(transformed_test) == "white_result")]))
# Predictions on the testing data
test_preds_prob <- predict(final_model, dtest)
test_preds <- ifelse(test_preds_prob > 0.5, 1, 0)  # Convert probabilities to binary class output

# Training data confusion matrix
confusionMatrix(factor(train_preds, levels = c(0, 1)),
                factor(as.numeric(transformed_train$white_result) - 1, levels = c(0, 1)))

# Testing data confusion matrix
confusionMatrix(factor(test_preds, levels = c(0, 1)),
                factor(as.numeric(transformed_test$white_result) - 1, levels = c(0, 1)))

