library(shiny)
library(shinythemes)
library(shinycssloaders)
library(tidyverse)
library(ggExtra)
library(data.table)
library(caret)
library(tidymodels)
library(tidyverse)
library(randomForest)
library(xgboost)
library(keras)
library(vip)
library(pROC)
library(dplyr)
library(MLmetrics)

# be able to upload up to 100MB size file
options(shiny.maxRequestSize = 100 * 1024^2)

options(shiny.legacy.datatable = TRUE)

data_initial <- read.csv("modified_games_data.csv", header = TRUE)


# Define UI for application

ui <- fluidPage(
  
  titlePanel("Predicting Chess Wins"),
  
  navbarPage(
    
    title = ("Group 10"),
    
    theme = shinytheme("flatly"),
    
    tabPanel("Home", icon = icon("info-circle"),
             
             titlePanel("Overview: User Instructions"),
             
             mainPanel(
               
               helpText("STAT 3106: Applied Machine Learning - Final Project for Group 10"),
               
               "Welcome to Group 10's Final Project. Authors: Aimee Oh, Phebe Lew, Sean Le Van"
               )
             
    ),
    
    tabPanel("Data", icon = icon("folder-open"),
             
             titlePanel("Upload Data"),
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 selectInput("dataset", "Dataset:", choices = c("Chess Club Games Data", "Upload your own file")),
                 
                 conditionalPanel(condition = "input.dataset == 'Upload your own file'",
                                  
                                  fileInput("file", "Select your files:",
                                            
                                            accept=c("text/csv",
                                                     
                                                     "text/comma-separated-values,text/plain",
                                                     
                                                     ".csv"))  
                                  
                 )
                 
               ),
               
               mainPanel(
                 
                 
                 dataTableOutput("data_preview")
                 
               )
               
             )
             
    ),
    
    tabPanel("EDA",
             
             titlePanel("Scatterplot"),
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 selectInput("response", "Response Variable (Y)", choices = NULL), 
                 
                 selectInput("explanatory", "Explanatory Variable (X)", choices = NULL),
                 
                 sliderInput("shade", "Transaparency Rate", min = 0, max = 1, value = 0.5, step = 0.1),
                 
                 checkboxInput("marginal", "Marginal Distributions", value = FALSE)
                 
               ),
               
               mainPanel(
                 
                 tabsetPanel(
                   
                   tabPanel("Scatterplot", 
                            
                            plotOutput("plot1")),
                   
                   tabPanel("Numeric Summary",
                            
                            dataTableOutput("result1"))
                   
                 )
                 
               )
               
               
             )
             
             
             
    ),
    
    tabPanel("Data Pre-Processing",
             
             titlePanel("Preprocessing"),
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 checkboxGroupInput("column_choices", label = "Select Columns to exclude:",
                                    
                                    choices = NULL),
                 
                 actionButton("data_clean", "Submit")
                 
               ),
               
               mainPanel(
                 
                 dataTableOutput("modified_preview")
                   
                 )
                 
               )
               
               
    ),
             
    tabPanel("Model",
             
             titlePanel("Random Forest Model"),
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 selectInput("model_type", "Model to train", choices = c("Random Forest", "Gradient Boosting Machine")),
                 
                 selectInput("type", "Model Type", choices = c("regression", "classification")),
                 
                 uiOutput("rf_select_nodes_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Random Forest",
                   
                   
                 ),
                 
                 uiOutput("rf_select_mtry_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Random Forest",
                   
                   
                 ),
                 
                 uiOutput("xg_select_nrounds_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Gradient Boosting Machine",
                   
                   
                 ),
                 
                 uiOutput("xg_select_maxdepth_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Gradient Boosting Machine",
                   
                   
                 ),
                 
                 uiOutput("xg_select_eta_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Gradient Boosting Machine",
                   
                   
                 ),
                 
                 uiOutput("xg_select_minchildw_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Gradient Boosting Machine",
                   
                   
                 ),
                 
                 uiOutput("xg_select_subsample_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.model_type == Gradient Boosting Machine",
                   
                   
                 ),
                 
                 selectInput("data_split", "Data Split Ratio", choices = c(0.6, 0.65, 0.7, 0.75, 0.8)),
                 
                 uiOutput("dynamic_select_ui"),
                 
                 conditionalPanel(
                   
                   condition = "input.type == classification",
                   
                   
                 ),
                 
                 actionButton("train", "Train Model")
                 
               ), 
               
               mainPanel(
                 
                 tabsetPanel(
                   
                   tabPanel("Model Train Set Output",
                            
                            verbatimTextOutput("model_train_output"),
                            
                            plotOutput("train_plot")),
                   
                   tabPanel("Model Test Set Output",
                            
                            verbatimTextOutput("model_test_output"),
                            
                            plotOutput("test_plot"))
                   
                 ),
                 
               )
               
             )
             
    )
             
    ),
    
  )


# Define Server

server <- function(input, output, session) {
  
  ##  
  
  File <- reactive({
    
    if(input$dataset == 'Upload your own file'){
      
      req(input$file)
      
      File <- input$file
      
      df <- data.frame(rbindlist(lapply(File$datapath, fread), use.names = TRUE, fill = TRUE))
      
      return(df)
      
    } else {
      
      return(data_initial)
    } 
    
  })
  
  
  ##
  
  observeEvent(File(), {
    
    updateSelectInput(session, "response",
                      
                      choices = names(File()))
  })
  
  
  
  observeEvent(File(), {
    
    updateSelectInput(session, "explanatory",
                      
                      choices = names(File()))
  }) 
  
  
  observeEvent(File(), {
    
    updateSelectInput(session, "var",
                      
                      choices = names(File()))
  })
  
  ##
  
  output$dynamic_select_ui <- renderUI({
    
    if(input$type == "classification"){
      
      checkboxGroupInput("target_choices", label = "Select Options (Choose Two) for Classification:",
                         
                         choices = NULL)
      
    }
    
  })
  
  ##
  
  observeEvent(input$type, {
    
    updateCheckboxGroupInput(session, "target_choices", choices = unique(File()[[input$response]]))
    
  })
  
  ##
  
  output$rf_select_nodes_ui <- renderUI({
    
    if (input$model_type == "Random Forest") {
      
      numericInput("nodes", "Number of Nodes per Tree:", value = 9)
    }
    
  })
  
  ##
  
  output$rf_select_mtry_ui <- renderUI({
    
    if (input$model_type == "Random Forest") {
      
      numericInput("mtry", "Number of Variables to Sample (mtry):", value = 3)
    }
    
  })
  
  ##
  
  output$xg_select_nrounds_ui <- renderUI({
    
    if (input$model_type == "Gradient Boosting Machine") {
      
      selectInput("nrounds", "Number of Iterations (nrounds):", choices = c(100,200,300))
    }
    
  })
  
  ##
  
  output$xg_select_maxdepth_ui <- renderUI({
    
    if (input$model_type == "Gradient Boosting Machine") {
      
      numericInput("maxdepth", "Maximum Depth of Tree:", value = 4)
    }
    
  })
  
  ##
  
  output$xg_select_eta_ui <- renderUI({
    
    if (input$model_type == "Gradient Boosting Machine") {
      
      selectInput("eta", "Learning Rate:", c(0.05, 0.1, 0.2, 0.3))
    }
    
  })
  
  ##
  
  output$xg_select_minchildw_ui <- renderUI({
    
    if (input$model_type == "Gradient Boosting Machine") {
      
      selectInput("minchildw", "Minimum Child Weight:", choices = c(0.05, 0.1, 0.2, 0.3))
    }
    
  })
  
  ##
  
  output$xg_select_subsample_ui <- renderUI({
    
    if (input$model_type == "Gradient Boosting Machine") {
      
      selectInput("subsample", "Subsampling ratio:", choices = c(0.4, 0.6))
    }
    
  })
  
  ##
  
  observeEvent(input$response, {
    
    updateCheckboxGroupInput(session, "column_choices", choices = colnames(File()))
    
  })
  
  ##
  
  output$modified_preview <- renderDataTable({
    
    data()
    
  }) 
  
  
  ##
  
  
  output$data_preview <- renderDataTable({
    
    File()
    
  }) 
  
  
  ##
  output$plot1 <- renderPlot({
    
    p = ggplot(data = File(), aes_string(x = input$explanatory, y = input$response)) +
      
      geom_point(alpha = input$shade) +
      
      theme_minimal() 
    
    
    if(input$marginal) {
      
      p <- ggMarginal(p, type = "histogram")
    }
    
    
    p
    
  })
  
  
  ##
  
  output$result1 <- renderDataTable({
    
    summary_data <- summary(File()[[input$response]])
    
    data.frame(Measure = names(summary_data), Value = as.character(summary_data))
    
  })
  
  ##
  
  data <- observeEvent(input$data_clean, {
    
    selected_columns <- input$column_choices
    df <- File()
    
    # Check if any columns are selected
    if (length(selected_columns) > 0) {
      
      # Subset the original data frame to exclude selected columns
      df <- df[, !colnames(df) %in% selected_columns, drop = FALSE]
      
      # Use df for further processing or update the UI
      # For example, you can render the modified data frame using renderDataTable()
      output$modified_preview <- renderDataTable({
        df
      })
      
    } else {
      # If no columns are selected, use the original data frame
      output$modified_preview <- renderDataTable({
        df
      })
    }
    
    return(df)
  })
  
  ##
  
  # render RF_train_plot
  
  ##
  
  # Train random forest model
  observeEvent(input$train, {
    browser()
    
    set.seed(1)
    
    if (input$model_type == "Random Forest"){
      if (input$type == "classification") {
        # getting target variable input by user
        target <- input$response
        df <- File()
        targetdf <- df[[target]]
        
        # changing target variable into binary classification
        df <- df %>%
          mutate(targetdf = ifelse(targetdf == input$target_choices[1], input$target_choices[1], input$target_choices[2]))
        df[[target]] <- as.factor(df[['targetdf']])
        
        # ensure factor levels are valid names
        levels(df[[target]]) <- make.names(levels(df[[target]]))
        
        df <- df[, !names(df) %in% c('targetdf')]
        
        # making sure the split ratio chosen by user is numeric
        data_split <- as.numeric(input$data_split)
        
        # splitting data
        index <- createDataPartition(df[[target]], p = data_split, list = FALSE)
        before_train <- df[index, ]
        before_test <- df[-index, ]
        
        target_formula <- as.formula(paste(input$response, "~ ."))
        
        # standardizing and normalizing data
        blueprint <- recipe(target_formula, data = before_train) %>%
          step_string2factor(all_nominal_predictors()) %>%
          step_nzv(all_predictors()) %>%
          step_impute_knn(all_predictors()) %>%
          step_center(all_numeric_predictors()) %>%
          step_scale(all_numeric_predictors()) %>%
          step_other(all_nominal(), threshold = 0.01, other = "other") %>%
          step_dummy(all_nominal_predictors())
        
        print(blueprint)
        
        blueprint_prep <- prep(blueprint, training = before_train)
        
        transformed_train <- bake(blueprint_prep, new_data = before_train)
        transformed_test <- bake(blueprint_prep, new_data = before_test)
        
        # Check the class of transformed_train
        
        print(class(transformed_train))
        print(dim(transformed_train))  # Check dimensions
        
        hyperparameters <- data.frame(.mtry = input$mtry,
                                      .min.node.size = input$nodes,
                                      .splitrule = "extratrees")
        
        print(class(hyperparameters))
        
        fitControl_final <- trainControl(method = "cv",
                                         number = 5,
                                         classProbs = TRUE,
                                         summaryFunction = twoClassSummary)
        
        print("after hyperparameters")
        
        mtry <- as.numeric(input$mtry)
        nodes <- as.numeric(input$nodes)
        
        RF_final <- train(target_formula, 
                          data = transformed_train,
                          method = "ranger",
                          trControl = fitControl_final,
                          metric = "ROC",
                          tuneGrid = data.frame(mtry = mtry,
                                                min.node.size = nodes,
                                                splitrule = "gini"),
                          num.trees = 300)
        
        RF_pred_train <- predict(RF_final, newdata = transformed_train)
        
        RF_train_results <- confusionMatrix(transformed_train[[target]], RF_pred_train)
        
        RF_Kappa <- RF_train_results$overall["Kappa"]
        
        RF_pred_test <- predict(RF_final, newdata = transformed_test)
        
        RF_test_results <- confusionMatrix(transformed_test[[target]], RF_pred_test)
        
        output$model_train_output <- renderPrint({
          print(RF_train_results)
        })
        
        output$model_test_output <- renderPrint({
          print(RF_test_results)
        })
        
        
      } else if (input$model_type == "regression") {
        print("regression")
        
        # getting target variable input by user
        target <- input$response
        df <- File()
        
        # making sure the split ratio chosen by user is numeric
        data_split <- as.numeric(input$data_split)
        
        # splitting data
        index <- createDataPartition(df[[target]], p = data_split, list = FALSE)
        before_train <- df[index, ]
        before_test <- df[-index, ]
        
        target_formula <- as.formula(paste(input$response, "~ ."))
        
        # standardizing and normalizing data
        blueprint <- recipe(target_formula, data = before_train) %>%
          step_string2factor(all_nominal_predictors()) %>%
          step_nzv(all_predictors()) %>%
          step_impute_knn(all_predictors()) %>%
          step_center(all_numeric_predictors()) %>%
          step_scale(all_numeric_predictors()) %>%
          step_other(all_nominal(), threshold = 0.01, other = "other") %>%
          step_dummy(all_nominal_predictors()) %>%
          step_log(-all_numeric_predictors() %>% nearZeroVar()) %>%
          step_naomit(everything())
        
        print(blueprint)
        
        blueprint_prep <- prep(blueprint, training = before_train)
        
        transformed_train <- bake(blueprint_prep, new_data = before_train)
        transformed_test <- bake(blueprint_prep, new_data = before_test)
        
        fitControl_final <- trainControl(method = "cv",
                                         number = 5,
                                         verboseIter = TRUE,
                                         returnData = FALSE,
                                         returnResamp = "final",
                                         classProbs = FALSE)
        
        mtry <- as.numeric(input$mtry)
        nodes <- as.numeric(input$nodes)
        
        RF_final <- train(target_formula, 
                          data = transformed_train,
                          method = "rf",
                          trControl = fitControl_final,
                          metric = "RMSE",
                          tuneGrid = expand.grid(mtry = mtry),
                          importance = TRUE,
                          ntree = 300)
        
        print("fit model")
        
        RF_pred_train <- predict(RF_final, newdata = transformed_train)
        
        RF_train_rmse <- sqrt(mean((transformed_train[[target]] - RF_pred_train)^2))
        
        print(RF_train_rmse)
        
        
        RF_pred_test <- predict(RF_final, newdata = transformed_test)
        
        print((transformed_test[[target]] - RF_pred_test)^2)
        
        RF_test_rmse <- sqrt(mean((transformed_test[[target]] - RF_pred_test)^2))
        
        print(RF_test_rmse)
        
        output$model_train_output <- renderPrint({
          print(paste("RMSE:", RF_train_rmse))
        })
        
        output$train_plot <- renderPlot({
          
          prediction_df <- data.frame(Actual = transformed_train[[target]], Predicted = RF_pred_train)
          
          graph <- ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
            geom_point(color = "blue") +  # Scatter plot of actual vs. predicted values
            geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Add a diagonal line for reference
            labs(x = "Actual", y = "Predicted", title = "Actual vs. Predicted Values") +  # Axis labels and plot title
            theme_minimal()  # Set plot theme
          
          graph
          
        })
        
        output$model_test_output <- renderPrint({
          print(paste("RMSE:", RF_test_rmse))
          
          prediction_df <- data.frame(Actual = transformed_test[[target]], Predicted = RF_pred_test)
          
          ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
            geom_point(color = "blue") +  # Scatter plot of actual vs. predicted values
            geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Add a diagonal line for reference
            labs(x = "Actual", y = "Predicted", title = "Actual vs. Predicted Values") +  # Axis labels and plot title
            theme_minimal()  # Set plot theme
        })
        
        output$test_plot <- renderPlot({
          
          prediction_df <- data.frame(Actual = transformed_test[[target]], Predicted = RF_pred_test)
          
          graph <- ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
            geom_point(color = "blue") +  # Scatter plot of actual vs. predicted values
            geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Add a diagonal line for reference
            labs(x = "Actual", y = "Predicted", title = "Actual vs. Predicted Values") +  # Axis labels and plot title
            theme_minimal()  # Set plot theme
          
          graph
          
        })
        
      }
    }
    else if(input$model_type == "Gradient Boosting Machine"){
      if (input$type == "classification") {
        # getting target variable input by user
        target <- input$response
        df <- File()
        targetdf <- df[[target]]
        
        # changing target variable into binary classification
        df <- df %>%
          mutate(targetdf = ifelse(targetdf == input$target_choices[1], input$target_choices[1], input$target_choices[2]))
        df[[target]] <- as.factor(df[['targetdf']])
        
        print(dim(df[[target]]))
        
        # ensure factor levels are valid names
        levels(df[[target]]) <- make.names(levels(df[[target]]))
        
        df <- df[, !names(df) %in% c('targetdf')]
        
        # making sure the split ratio chosen by user is numeric
        data_split <- as.numeric(input$data_split)
        
        # splitting data
        index <- createDataPartition(df[[target]], p = data_split, list = FALSE)
        before_train <- df[index, ]
        before_test <- df[-index, ]
        
        target_formula <- as.formula(paste(input$response, "~ ."))
        
        # standardizing and normalizing data
        blueprint <- recipe(target_formula, data = before_train) %>%
          step_string2factor(all_nominal_predictors()) %>%
          step_nzv(all_predictors()) %>%
          step_impute_knn(all_predictors()) %>%
          step_center(all_numeric_predictors()) %>%
          step_scale(all_numeric_predictors()) %>%
          step_other(all_nominal(), threshold = 0.01, other = "other") %>%
          step_dummy(all_nominal_predictors())
        
        print(blueprint)
        
        blueprint_prep <- prep(blueprint, training = before_train)
        
        transformed_train <- bake(blueprint_prep, new_data = before_train)
        transformed_test <- bake(blueprint_prep, new_data = before_test)
        
        # Check the class of transformed_train
        
        print(class(transformed_train))
        print(dim(transformed_train))  # Check dimensions
        
        fitControl_final <- trainControl(method = "none",
                                         classProbs = TRUE)
        
        print("after hyperparameters")
        
        XG_final <- train(target_formula, 
                          data = transformed_train,
                          method = "xgbTree",
                          trControl = fitControl_final,
                          metric = "ROC",
                          tuneGrid = expand.grid(nrounds = input$nrounds,   
                                                 max_depth = input$maxdepth, 
                                                 eta = input$eta,    
                                                 min_child_weight = input$minchildw, 
                                                 subsample = input$subsample, 
                                                 gamma = 0,
                                                 colsample_bytree = 1))
        
        print("fit model")
        
        XG_pred_train <- predict(XG_final, newdata = transformed_train)
        
        XG_train_results <- confusionMatrix(transformed_train[[target]], XG_pred_train)
        
        print(XG_train_results)
        
        XG_Kappa <- XG_train_results$overall["Kappa"]
        
        XG_pred_test <- predict(XG_final, newdata = transformed_test)
        
        XG_test_results <- confusionMatrix(transformed_test[[target]], XG_pred_test)
        
        print(XG_test_results)
        
        output$model_train_output <- renderPrint({
          print(XG_train_results)
        })
        
        output$model_test_output <- renderPrint({
          print(XG_test_results)
        })
        
        
      } else if (input$model_type == "regression") {
        print("regression")
        
        # getting target variable input by user
        target <- input$response
        df <- File()
        
        # making sure the split ratio chosen by user is numeric
        data_split <- as.numeric(input$data_split)
        
        # splitting data
        index <- createDataPartition(df[[target]], p = data_split, list = FALSE)
        before_train <- df[index, ]
        before_test <- df[-index, ]
        
        target_formula <- as.formula(paste(input$response, "~ ."))
        
        # standardizing and normalizing data
        blueprint <- recipe(target_formula, data = before_train) %>%
          step_string2factor(all_nominal_predictors()) %>%
          step_nzv(all_predictors()) %>%
          step_impute_knn(all_predictors()) %>%
          step_center(all_numeric_predictors()) %>%
          step_scale(all_numeric_predictors()) %>%
          step_other(all_nominal(), threshold = 0.01, other = "other") %>%
          step_dummy(all_nominal_predictors()) %>%
          step_log(-all_numeric_predictors() %>% nearZeroVar()) %>%
          step_naomit(everything())
        
        print(blueprint)
        
        blueprint_prep <- prep(blueprint, training = before_train)
        
        transformed_train <- bake(blueprint_prep, new_data = before_train)
        transformed_test <- bake(blueprint_prep, new_data = before_test)
        
        fitControl_final <- trainControl(method = "none",
                                         classProbs = FALSE)
        
        XG_final <- train(target_formula, 
                          data = transformed_train,
                          method = "xgbTree",
                          trControl = fitControl_final,
                          metric = "RMSE",
                          tuneGrid = expand.grid(nrounds = input$nrounds,   
                                                 max_depth = input$maxdepth, 
                                                 eta = input$eta,    
                                                 min_child_weight = input$minchildw, 
                                                 subsample = input$subsample, 
                                                 gamma = 0,
                                                 colsample_bytree = 1))
        
        print("fit model")
        
        XG_pred_train <- predict(XG_final, newdata = transformed_train)
        
        XG_train_rmse <- sqrt(mean((transformed_train[[target]] - XG_pred_train)^2))
        
        print(XG_train_rmse)
        
        
        XG_pred_test <- predict(XG_final, newdata = transformed_test)
      
        print((transformed_test[[target]] - XG_pred_test)^2)
        
        XG_test_rmse <- sqrt(mean((transformed_test[[target]] - XG_pred_test)^2))
        
        print(XG_test_rmse)
        
        output$model_train_output <- renderPrint({
          print(paste("RMSE:", XG_train_rmse))
        })
        
        output$train_plot <- renderPlot({
          
          prediction_df <- data.frame(Actual = transformed_train[[target]], Predicted = XG_pred_train)
          
          graph <- ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
            geom_point(color = "blue") +  # Scatter plot of actual vs. predicted values
            geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Add a diagonal line for reference
            labs(x = "Actual", y = "Predicted", title = "Actual vs. Predicted Values") +  # Axis labels and plot title
            theme_minimal()  # Set plot theme
          
          graph
          
        })
        
        output$model_test_output <- renderPrint({
          print(paste("RMSE:", XG_test_rmse))
          
          prediction_df <- data.frame(Actual = transformed_test[[target]], Predicted = XG_pred_test)
          
          ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
            geom_point(color = "blue") +  # Scatter plot of actual vs. predicted values
            geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Add a diagonal line for reference
            labs(x = "Actual", y = "Predicted", title = "Actual vs. Predicted Values") +  # Axis labels and plot title
            theme_minimal()  # Set plot theme
        })
        
        output$test_plot <- renderPlot({
          
          prediction_df <- data.frame(Actual = transformed_test[[target]], Predicted = XG_pred_test)
          
          graph <- ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
            geom_point(color = "blue") +  # Scatter plot of actual vs. predicted values
            geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Add a diagonal line for reference
            labs(x = "Actual", y = "Predicted", title = "Actual vs. Predicted Values") +  # Axis labels and plot title
            theme_minimal()  # Set plot theme
          
          graph
          
        })
        
      }
    }
    
  })
  
}





# Run the application 

shinyApp(ui = ui, server = server)
