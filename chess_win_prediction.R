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

### PURPOSELY MESSING UP THE DATASET TO ALLOW FOR DATA PREPROCESSING
# I did this after data exploration revealed no NA values.
# I restricted the columns that I'm messing up.
# I think for the final submission, we could just do this in a separate
# appendix document.
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
    white_result %in% c("win", "threecheck", "kingofthehill") ~ 1,
    white_result %in% c("checkmated", "timeout", "resigned", "abandoned") ~ 0,
    TRUE ~ 0.5 # Assumes all other cases are draws
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
  step_dummy(all_nominal())

blueprint_prep <- prep(blueprint, training = games_train)

transformed_train <- bake(blueprint_prep, new_data = games_train)
transformed_test <- bake(blueprint_prep, new_data = games_test)

### NEXT STEPS (Is there anything I am missing?)
# 1. RF
# 2. XGBoost
# 3. SVM
# 4. Artificial Neural Network
# 5. Ensemble Method

