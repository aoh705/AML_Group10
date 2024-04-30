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
               
               helpText("STAT 3106: Applied Machine Learning - Final Project for Group 10"))
             
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
    
    
    
    
    tabPanel("Model",
             
             titlePanel("Random Forest Model"),
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 selectInput("rf_type", "Random Forest Model Type", choices = c("classification", "numeric")),
                 
                 numericInput("nodes", "Number of Nodes per Tree:", value = 9),
                 
                 numericInput("mtry", "Number of Variables to Sample:", value = 3),
                 
                 selectInput("data_split", "Data Split Ratio", choices = c(0.6, 0.65, 0.7, 0.75, 0.8)),
                 
                 checkboxGroupInput("target_choices", label = "Select Options (Choose Two) for Classification:",
                                    
                                    choices = NULL),
                 
                 actionButton("train", "Train Model")
                
               ), 
               
               mainPanel(
                 
                 tabsetPanel(
                   
                   #tabPanel("Random Forest Model",
                            
                            #plotOutput("plot2")),
                   
                   tabPanel("Random Forest Model Train Set Output",
                            
                            verbatimTextOutput("model_train_output")),
                   
                   tabPanel("Random Forest Model Test Set Output",
                            
                            verbatimTextOutput("model_test_output"))
                   
                 ),
                 
               )
               
             )
             
             
    )
    
    
    
  )
  
  
  
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
  
  observeEvent(input$response, {
    
    updateCheckboxGroupInput(session, "target_choices", choices = unique(File()[[input$response]]))
    
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
  
  #plot2 <- eventReactive(input$click, 
                         
                         #ggplot(data = File(), aes_string(x = input$var)) +
                           
                           #geom_histogram(binwidth = diff(range(File()[[input$var]]) / input$bins), fill = input$color, color = "black") +
                           
                           #labs(x = input$var, y = "Frequency", title = "Histogram") +
                           
                           #theme_minimal()
                         
  #)
  
  
  
  #output$plot2 <- renderPlot({
    
    #plot2() 
    
  #})
  
  ##
  
  # Train random forest model
  observeEvent(input$train, {
    browser()
    
    set.seed(1)
    
    if (input$rf_type == "classification") {
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
      
      fitControl_final <- trainControl(method = "none",
                                       classProbs = TRUE)
      
      print("after hyperparameters")
      
      print(class(input$mtry))
      print(class(input$nodes))
      
      mtry <- as.numeric(input$mtry)
      nodes <- as.numeric(input$nodes)
      
      RF_final <- train(target_formula, 
                        data = transformed_train,
                        method = "ranger",
                        trControl = fitControl_final,
                        metric = "ROC",
                        tuneGrid = data.frame(.mtry = mtry,
                                              .min.node.size = nodes,
                                              .splitrule = "extratrees"),
                        num.trees = 300)
      
      print("fit model")
      
      RF_pred_train <- predict(RF_final, newdata = transformed_train)
      
      RF_train_results <- confusionMatrix(transformed_train$white_result, RF_pred_train)
      
      print(RF_train_results)
      
      RF_Kappa <- RF_train_results$overall["Kappa"]
      
      RF_pred_test <- predict(RF_final, newdata = transformed_test)
      
      RF_test_results <- confusionMatrix(transformed_test$white_result, RF_pred_test)
      
      print(RF_test_results)
      
      output$model_train_output <- renderPrint({
        print(RF_train_results)
      })
      
      output$model_test_output <- renderPrint({
        print(RF_test_results)
      })
      
      
    } else if (input$rf_type == "numeric") {
      print("not classification")
    }
    
    # Generate predictions
    
    # Display visualizations and interpretations
  })
  
}





# Run the application 

shinyApp(ui = ui, server = server)
