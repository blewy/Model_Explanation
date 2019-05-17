library(mlbench)
library(tidyverse)
library(caret)
#devtools::install_github("laresbernardo/lares")
library(lares)
library(gbm)
library(mlr)

source("./scripts/aux_function.R")

#https://www.r-bloggers.com/dalex-and-h2o-machine-learning-model-interpretability-and-feature-explanation/

# Get data
insurance.data <- read.csv("./data/insurance.csv")
View(insurance.data)
str(insurance.data)
summarizeColumns(insurance.data)
summarizeLevels(insurance.data, cols = NULL)

#correct Names
names(insurance.data)  <- make.names(names(insurance.data))
#View(insurance.data)
id.data <- insurance.data$competitorname


# Basic density
ggplot(insurance.data,
       aes(
         x = charges,
         y = ..density..,
         color = sex,
         fill = sex
       ),
       legend = TRUE) +
  labs(title = "Charge amount") +
  geom_density(alpha = 0.2)  + geom_vline(
    data = insurance.data,
    aes(xintercept = mean(charges), colour = sex),
    linetype = "dashed",
    size = 0.5
  )  + theme_minimal()


# Done
table(insurance.data$sex)
prop.table(table(insurance.data$sex))



insurance.data.model <-
  as.data.frame(model.matrix( ~ . - 1, insurance.data))
insurance.data.model$sexfemale <- NULL
insurance.data.model$sexmale <-
  as.factor(insurance.data.model$sexmale)
#View(insurance.data.model)

#Split Data
set.seed(1951)
trainIndex <- createDataPartition(
  insurance.data.model$sexmale,
  p = 0.7,
  #Proportion of training data
  list = FALSE,
  times = 1
)

n = nrow(insurance.data.model)
id <- 1:n
training = insurance.data.model[trainIndex , ]
id_training <- id[trainIndex]
testing = insurance.data.model[-trainIndex , ]
id_testing <- id[-trainIndex]


#look at variable classes
split(names(insurance.data), sapply(insurance.data, function(x) {
  class(x)
}))
#splitting the data based on class
split(names(insurance.data.model),
      sapply(insurance.data.model, function(x) {
        class(x)
      }))
#View(insurance.data.model)

#create a task
trainTask <-
  makeClassifTask(data =  training,
                  target = "sexmale",
                  positive = 1)
#create a task
testnTask <- makeClassifTask(data =  testing, target = "sexmale")

#Xgboost
#load xgboost
set.seed(1001)
getParamSet("classif.xgboost")

#make learner with inital parameters
xg_set <- makeLearner("classif.xgboost", predict.type = "prob")
xg_set$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  #nrounds = 500,
  early_stopping_rounds = 20,
  print_every_n = 10
  
)

#define parameters for tuning
xg_ps <- makeParamSet(
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  makeIntegerParam("max_depth", lower = 3, upper = 20),
  makeNumericParam("lambda", lower = 0.55, upper = 0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight", lower = 1, upper = 5),
  makeNumericParam("colsample_bytree", lower = 0.2, upper = 0.8)
)

#define search function
#rancontrol <- makeTuneControlMBO(budget = 10) #do 100 iterations
rancontrol <- makeTuneControlRandom(maxit = 25L) #do 100 iterations


#5 fold cross validation
set_cv <- makeResampleDesc("CV", iters = 3L, stratify = TRUE)

#tune parameters
xg_tune <- tuneParams(
  learner = xg_set,
  task = trainTask,
  resampling = set_cv,
  measures = auc,
  par.set = xg_ps,
  control = rancontrol
)

data = generateHyperParsEffectData(
  xg_tune ,
  include.diagnostics = FALSE,
  trafo = FALSE,
  partial.dep = TRUE
)

#cross validation results
data$data

View(data$data)

#set parameters
xg_new <- setHyperPars(learner = xg_set, par.vals = xg_tune$x)

#train model with the best parameters
xgmodel <- train(xg_new, trainTask)

#predict on test data
predict.xg <- predict(xgmodel, testnTask)


#using MLR
calculateConfusionMatrix(predict.xg)
calculateROCMeasures(predict.xg)

#Using caret
confusionMatrix(as.factor(as.integer(predict.xg$data$prob.1 > 0.5)), predict.xg$data$truth)


#Calculo do AUC com o package pROC
library(pROC)
auc(roc(as.numeric(predict.xg$data$truth), predict.xg$data$prob.1))


#matrix de confusão
data.plot.test.xgboost <-
  data.frame(
    obs = as.factor(if_else(predict.xg$data$truth == 1, "Yes", "No")),
    predicted = predict.xg$data$prob.1,
    orig = "XGBoost"
  )
plot_confusion_matrix(data.plot.test.xgboost,
                      threshold = 0.5,
                      sSubtitle = "XGBoost Model")

data.plot.test.xgboost <-
  data.frame(
    obs = predict.xg$data$truth == 1,
    predicted = predict.xg$data$prob.1,
    orig = "XGBoost"
  )
#Gráficos do AUC e tabela com informação do indicadores para cada threshold
roc <- calculate_roc(data.plot.test.xgboost, 1, 1, n = 100)
mincost <- min(roc$cost)
roc %>%
  mutate(auc = ifelse(
    cost == mincost,
    cell_spec(
      sprintf("%.5f", auc),
      "html",
      color = "green",
      background = "lightblue",
      bold = T
    ),
    cell_spec(
      sprintf("%.5f", auc),
      "html",
      color = "black",
      bold = F
    )
  )) %>%
  kable("html", escape = F, align = "c") %>%
  kable_styling(
    bootstrap_options = "striped",
    full_width = F,
    position = "center"
  ) %>%
  scroll_box(height = "600px")

#seleccionar o threshold optimo baseado no custo
threshold = 0.61404
plot_roc(roc, threshold, cost_of_fp = 1, cost_of_fn = 10)


# Begin explanations
#--------------  Lime package ----------------

library(lime)
test.data <- testing
row.names(test.data) <- id_testing # Add ID
names(test.data)
test.data[, "sexmale"] <- NULL # remove target

explainer <- lime(
  test.data ,
  xgmodel,
  bin_continuous = TRUE,
  n_bins = 5,
  quantile_bins = FALSE
)

save(explainer, file = "./models/explainer_classification.rda")

explanation <- explain(
  test.data[2,],
  explainer,
  n_labels = 1,
  n_features = 6,
  kernel_width = 0.5,
  feature_select = "highest_weights"
)
explanation[, 1:6]

plot_features(explanation, ncol = 2)
plot_explanations(explanation)


#--------------     DALEX package  ----------------

#https://rawgit.com/pbiecek/DALEX_docs/master/vignettes/DALEX_caret.html#3_classification_use_case_-_wine_data


library(DALEX)

#"Help function for MLR predictions"
custom_predict <- function(object, newdata) {
  pred <- predict(object, newdata = newdata)
  response <- (pred$data$prob.1)
  return(response)
}

explainer_gbm <- DALEX::explain(
  xgmodel,
  label = "xgmodel",
  data = testing,
  y = as.numeric(testing$sexmale),
  predict_function = custom_predict
)

mp_gbm <- model_performance(explainer_gbm)


# Variables Importance
vi_classif_gbm <- variable_importance(explainer_gbm)
plot(vi_classif_gbm)



# Partial Dependence plot
pdp_classif_gbm  <-
  variable_response(explainer_gbm, variable = "bmi", type = "pdp")
plot(pdp_classif_gbm)


# Acumulated Local Effects plot
ale_classif_gbm <-
  variable_response(explainer_gbm, variable = "age", type = "ale")
plot(ale_classif_gbm)

# Acumulated Local Effects plot
ale_classif_gbm <-
  variable_response(explainer_gbm, variable = "children", type = "ale")
plot(ale_classif_gbm)


#explain Observation using breakdown
observation_explain <-
  prediction_breakdown(explainer_gbm, observation = testing[2,-2],
                       direction = "up")
plot(observation_explain)

pred <- predict(xgmodel, testnTask)
pred$data[2,]

#-------------------------- Live package -------------------------------

library(live)
library(mlr)
similar <- sample_locally(
  data = testing,
  explained_instance = testing[2,],
  explained_var = "sexmale",
  size = 50
)

similar1 <- add_predictions(
  to_explain = similar,
  black_box_model = xgmodel,
  predict_fun = custom_predict
)

trained <- fit_explanation(live_object = similar1,
                           white_box = "regr.lm",
                           selection = FALSE)


plot(trained,  type = "waterfallplot", direction = "up")
plot(trained,  type = "forestplot", direction = "up")


#---------------------- BreakDown package ----------------------------
#https://pbiecek.github.io/breakDown/articles/break_caret.html


library(breakDown)
observation_explain <-
  broken(
    xgmodel,
    testing[2,],
    data = training,
    predict.function = custom_predict,
    direction = "down",
    keep_distributions = TRUE
  )

observation_explain

plot(observation_explain,
     direction = "down",
     top_features = 6) + ggtitle("breakDown plot for caret/XGB model")




# ------------------------  Ceteris Paribus --------------------------

#https://pbiecek.github.io/ceterisParibus/articles/coral.html
#https://pbiecek.github.io/ceterisParibus/articles/ceteris_paribus.html
library("DALEX")
library("ceterisParibus")

explainer_gbm <- DALEX::explain(
  xgmodel,
  label = "xgb",
  data = testing[, -2],
  y = testing$sexmale,
  predict_function = custom_predict
)

#https://pbiecek.github.io/ceterisParibus/articles/ceteris_paribus.html#cheatsheet
#file:///D:/Users/f003985/Downloads/CeterisParibusCheatsheet.pdf

cr_rf  <- ceteris_paribus(explainer_gbm, testing[2, -2])

plot(
  cr_rf,
  plot_residuals = FALSE,
  selected_variables = c("age", "bmi", "children")
)
plot(
  cr_rf,
  plot_residuals = TRUE,
  selected_variables = c("age", "bmi", "children")
)


#Local model FIT
neighbours <- select_neighbours(testing, testing[2, ], n = 20)
cr_rf  <- ceteris_paribus(explainer_gbm, neighbours)
plot(
  cr_rf,
  show_profiles = TRUE,
  show_observations = TRUE,
  show_rugs = TRUE,
  plot_residuals = TRUE,
  selected_variables = c("age", "bmi")
)
#plot(cr_rf)

#average model Response
cp_rf <- ceteris_paribus(explainer_gbm,
                         testing[, -2], y = testing$charges)

plot(
  cp_rf,
  selected_variables = c("age", "bmi"),
  aggregate_profiles = mean,
  show_observations = FALSE
)


# ------------------------ IML ---------------------------------

library("iml")

#"Help function for MLR predictions"
custom_predict2 <- function(object, newdata) {
  pred <- predict(object, newdata = newdata)
  response <- (pred$data$prob.1)
  return(response)
}

X = testing[which(names(testing) != "sexmale")]
predictor = Predictor$new(
  xgmodel,
  data = X,
  y = as.numeric(testing$sexmale),
  predict.fun = custom_predict2
)

#Feature importance
imp = FeatureImp$new(predictor, loss = "ce")
plot(imp)


#Feature behavior  - PDP Plot
#age
ale = FeatureEffect$new(predictor, feature = "age")
ale$plot()

#bmi
ale = FeatureEffect$new(predictor, feature = "bmi")
ale$plot()

#children
ale = FeatureEffect$new(predictor, feature = "children")
ale$plot()

#all
effs = FeatureEffects$new(predictor)
plot(effs)


#Measure interactions
interact = Interaction$new(predictor)
plot(interact)

#two way interaction
interact = Interaction$new(predictor, feature = "charges")
plot(interact)


#Surrogate model + replace model with another interpretable one
tree = TreeSurrogate$new(predictor, maxdepth = 3)
plot(tree)


#Explain single predictions with a local model
lime.explain = LocalModel$new(predictor, x.interest = X[2, ])
plot(lime.explain)


#Explain single predictions with game theory
#Shapley value
shapley = Shapley$new(predictor, x.interest = X[2, ])
shapley$plot()

results = shapley$results
head(results)

