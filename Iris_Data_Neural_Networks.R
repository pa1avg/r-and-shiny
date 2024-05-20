library('shiny')
library(neuralnet)
library(datasets)

iris <- iris

iris$setosa <- iris$Species == "setosa"
iris$virginica <- iris$Species == "virginica"
iris$versicolor <- iris$Species == "versicolor"

train_idx <- sample(nrow(iris), 3/4 * nrow(iris))
data_train <- iris[train_idx, ]
data_test <- iris[-train_idx, ]

Y_options <- c("all" ,"setosa", "virginica", "versicolor")

ui <- fluidPage(
  fluidRow(wellPanel(
    h1('Neural Network Classification on Iris Data'),
    h3('c23102378')
  )),
  fluidRow(
    column(4,wellPanel(
      selectInput("species", label = h3("Species"), 
                                 choices = list(Y_options = Y_options), 
                                 selected = 1))
           ),
    column(4, wellPanel(
      selectInput("act_func", label = h3("Activation Function"),
                  choices = list("tanh", "relu", "logistic"),
                  selected = 1))
          ),
    
    column(4, wellPanel(
      sliderInput("max_steps", label = h3("Maximum number of steps"),
                  min = 10000, max = 100000, step = 100, value = 20000))
    ),
    
    column(3, wellPanel(
      sliderInput("threshold", label = h3("Threshold"),
                  min = 0.001, max = 0.5, step = 0.001, value = 0.1))
    ),
    
    column(3, wellPanel(
      numericInput("epochs", label = h3("Number of repetitions"),
                   value = 5, max = 20)
    )),
    
    column(3, wellPanel(
      numericInput("layer1", label = h3("Nodes in Layer 1"),
                   value = 5, max = 10)
    )),
    
    column(3, wellPanel(
      numericInput("layer2", label = h3("Nodes in Layer 2"),
                   value = 5, max = 10)
    ))
    
    ),
  
  actionButton("action", label = "Create Network"),
  
  hr(),
  fluidRow(column(2, verbatimTextOutput("value"))),

  plotOutput(outputId = 'neuralnetwork'),
  plotOutput(outputId = 'heatmap')
)

server <- function(input, output){
  
  observeEvent(input$action,{
    
    chosen_species <- input$species
    act_func <- input$act_func
    threshold <- input$threshold
    max_steps <- input$max_steps
    epochs <- input$epochs
    layer1 <- input$layer1
    layer2 <- input$layer2
    
    if (chosen_species == "all"){
      formula <- paste("setosa + virginica + versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
      
    }
    else{
      formula <- paste(chosen_species, "~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
    }
    
    nn <- neuralnet(formula = formula, data_train, 
                    linear.output = FALSE, hidden = c(layer1, layer2), rep = epochs, 
                    act.fct = act_func, threshold = threshold, stepmax = max_steps)
    
    output$neuralnetwork <- renderPlot({
      plot(nn, rep = 'best')
      
    })
  })
  
  observeEvent(input$action,{
    
    chosen_species <- input$species
    act_func <- input$act_func
    threshold <- input$threshold
    max_steps <- input$max_steps
    epochs <- input$epochs
    layer1 <- input$layer1
    layer2 <- input$layer2
    
    if (chosen_species == "all"){
      formula <- paste("setosa + virginica + versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
      
    }
    else{
      formula <- paste(chosen_species, "~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
    }
    
    nn <- neuralnet(formula = formula, data_train, 
                    linear.output = FALSE, hidden = c(layer1, layer2), rep = epochs, 
                    act.fct = act_func, threshold = threshold, stepmax = max_steps)
    
    output$heatmap <- renderPlot({
      
      pred <- compute(nn, data_test[-5:-8])
      index <- apply(pred$net.result, 1, which.max)
      results <- c("setosa", "virginica", "versicolor")[index]
      
      predictions <- NULL
      
      if (chosen_species == "all"){
        predictions <- apply(predict(nn, data_test, type = class), 1, which.max)
      }
      else{
        predictions <- predict(nn, data_test, type = class) > 0.5
      }
      
      matrix <- as.matrix(table(data_test$Species, predictions))
      heatmap(matrix, scale="column", 
              main = paste("Heatmap for plants of", chosen_species,  "species"))
      #Error: 'x' must have at least 2 rows and 2 columns
      #This occurs when the neural network places all of its predictions into the same column
      #Thus no heatmap can be produced
    })
  })
}

shinyApp(ui = ui, server = server)