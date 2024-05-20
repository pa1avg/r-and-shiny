library(neuralnet)
library(datasets)

iris <- iris

iris$setosa <- iris$Species == "setosa"
iris$virginica <- iris$Species == "virginica"
iris$versicolor <- iris$Species == "versicolor"

train_idx <- sample(nrow(iris), 3/4 * nrow(iris))
data_train <- iris[train_idx, ]
data_test <- iris[-train_idx, ]

###########################################
#Neural net models

nn <- neuralnet(setosa + virginica + versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                data = data_train, hidden = c(5,5), rep = 5, err.fct = "ce", 
                linear.output = FALSE, lifesign = "minimal", stepmax = 1000000, threshold = 0.001)

plot(nn, rep = "best", show.weights = FALSE)
#pred <- predict(nn, data_test, type = class)
pred <- compute(nn, data_test[-5:-8])
index <- apply(pred$net.result, 1, which.max)
results <- c("setosa", "virginica", "versicolor")[index]
table(data_test$Species, results)

matrix <- as.matrix(table(data_test$Species, 
                          apply(predict(nn, data_test, type = class), 
                                1, which.max)))
heatmap(matrix, scale="column", main = "Heatmap of Predicted Species vs True Species")

table(data_test$Species, results)

###########################################
#Individual Species

nn_setosa <- neuralnet(setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                data = data_train, hidden = c(5,5), rep = 5, err.fct = "ce", 
                linear.output = FALSE, lifesign = "minimal", stepmax = 1000000, threshold = 0.001)

plot(nn_setosa, rep = "best", show.weights = FALSE)
#pred <- predict(nn, data_test, type = class)
pred_setosa <- compute(nn_setosa, data_test[-5:-8])
index_setosa <- apply(pred$net.result, 1, which.max)
results_setosa <- c("setosa", "virginica", "versicolor")[index]
table(data_test$Species, predict(nn_setosa, data_test, type = class) > 0.5)

matrix_setosa <- as.matrix(table(data_test$Species, predict(nn_setosa, data_test, type = class) > 0.5))
heatmap(matrix_setosa, scale="column", main = "Heatmap Predicting if a plant is a Setosa")

#

nn_virginica <- neuralnet(virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                       data = data_train, hidden = c(5,5), rep = 5, err.fct = "ce", 
                       linear.output = FALSE, lifesign = "minimal", stepmax = 1000000, threshold = 0.001)

plot(nn_virginica, rep = "best", show.weights = FALSE)
#pred <- predict(nn, data_test, type = class)
pred_virginica <- compute(nn_virginica, data_test[-5:-8])
index_virginica <- apply(pred$net.result, 1, which.max)
results_setosa <- c("setosa", "virginica", "versicolor")[index]
table(data_test$Species, predict(nn_virginica, data_test, type = class) > 0.5)

matrix_virginica <- as.matrix(table(data_test$Species, predict(nn_virginica, data_test, type = class) > 0.5))
heatmap(matrix_virginica, scale="column", main = "Heatmap Predicting if a plant is a Virginica")

#

nn_versicolor <- neuralnet(versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                         data = data_train, hidden = c(5,5), rep = 5, err.fct = "ce", 
                         linear.output = FALSE, lifesign = "minimal", stepmax = 1000000, threshold = 0.001)

plot(nn_versicolor, rep = "best", show.weights = FALSE)
#pred <- predict(nn, data_test, type = class)
pred_versicolor <- compute(nn_versicolor, data_test[-5:-8])
index_versicolor <- apply(pred$net.result, 1, which.max)
results_versicolor <- c("setosa", "virginica", "versicolor")[index]
table(data_test$Species, predict(nn_versicolor, data_test, type = class) > 0.5)

matrix_versicolor <- as.matrix(table(data_test$Species, predict(nn_versicolor, data_test, type = class) > 0.5))
heatmap(matrix_versicolor, scale="column", main = "Heatmap Predicting if a plant is a Versicolor")

##################################


nn <- neuralnet(setosa + virginica + versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                data = data_train, hidden = c(5,5), rep = 5, err.fct = "ce", 
                linear.output = FALSE, lifesign = "minimal", stepmax = 1000000, threshold = 0.001)

plot(nn, rep = "best")


nn_setosa <- neuralnet(setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                       data = data_train, hidden = c(5,5), rep = 5, err.fct = "ce", 
                       linear.output = FALSE, lifesign = "minimal", stepmax = 1000000, threshold = 0.001)

plot(nn_setosa, rep = "best")
