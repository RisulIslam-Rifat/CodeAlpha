# ============================
# Iris Flower Classification
# ============================

# 1. Load packages (install first if needed)
# install.packages("caret")
# install.packages("ggplot2")
library(caret)
library(ggplot2)

# 2. Load the dataset
# (Change the path if your file is somewhere else)
iris_df <- read.csv("Iris.csv", stringsAsFactors = TRUE)

# If there is an Id column (like in the common Kaggle file), drop it
if ("Id" %in% names(iris_df)) {
  iris_df$Id <- NULL
}

# Check structure
str(iris_df)

# 3. Exploratory plots (optional but helpful)
pairs(iris_df[, 1:4], col = iris_df$Species, pch = 19)

# 4. Train / Test split (80% train, 20% test)
set.seed(123)  # for reproducibility
train_index <- createDataPartition(iris_df$Species, p = 0.8, list = FALSE)

train_data <- iris_df[train_index, ]
test_data  <- iris_df[-train_index, ]

# 5. Train a machine learning model
# Here we use k-Nearest Neighbors (kNN) as an example classifier.
control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

set.seed(123)
knn_model <- train(
  Species ~ .,              # Species is the target; all other columns are predictors
  data = train_data,
  method = "knn",           # k-Nearest Neighbors algorithm
  trControl = control,
  preProcess = c("center", "scale"),  # standardize features
  tuneLength = 10           # try different k values
)

# View the best model and tuning results
print(knn_model)

# 6. Evaluate model on the test set
test_pred <- predict(knn_model, newdata = test_data)

conf_mat <- confusionMatrix(test_pred, test_data$Species)
print(conf_mat)

# Overall accuracy
cat("Test Accuracy:", conf_mat$overall["Accuracy"], "\n")

# 7. Basic classification concepts:
# - Inputs: sepal length/width, petal length/width
# - Output: species label (setosa, versicolor, virginica)
# - Model: kNN (classifies based on nearest neighbors in feature space)
# - Metric: accuracy + confusion matrix on held-out test data
