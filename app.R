
# R lab 10 - 04/16/24

# Building Web Applications with Shiny Package


library(shiny)

library(shinythemes)

library(shinycssloaders)

library(tidyverse)

library(ggExtra)

library(data.table)


data_initial <- read.csv("chess_games_data.csv", header = TRUE)


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
                 
                 selectInput("dataset", "Dataset:", choices = c("Club Games Data", "Upload your own file")),
                 
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
    
    tabPanel("Model",
             
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
    
    
    
    
    tabPanel("Fourth Panel",
             
             titlePanel("Histogram"),
             
             sidebarLayout(
               
               sidebarPanel(
                 
                 selectInput("var", "Variable", choices = NULL), 
                 
                 numericInput("bins", "Number of bins", min = 1, max = 50, step = 1, value = 10),
                 
                 radioButtons("color", "Color of bins:",
                              
                              choices = list("Blue" = "blue", "Red" = "red", "Green" = "green"),
                              
                              selected = "blue"),
                 
                 actionButton("click","Submit")
                 
               ),
               
               mainPanel(
                 
                 tabsetPanel(
                   
                   tabPanel("Histogram",
                            
                            plotOutput("plot2"))
                   
                   
                 )
                 
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
  
  plot2 <- eventReactive(input$click, 
                         
                         ggplot(data = File(), aes_string(x = input$var)) +
                           
                           geom_histogram(binwidth = diff(range(File()[[input$var]]) / input$bins), fill = input$color, color = "black") +
                           
                           labs(x = input$var, y = "Frequency", title = "Histogram") +
                           
                           theme_minimal()
                         
  )
  
  
  
  output$plot2 <- renderPlot({
    
    plot2() 
    
  })
  
}





# Run the application 

shinyApp(ui = ui, server = server)
