#library(MASS)
#library(mlbench)
library(tidyverse)
library(caret)
#library(mlr)
#devtools::install_github("laresbernardo/lares")
library(lares)
library(gbm)
library(mlr)

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


#get an id for large claims, in the future a model will be develop to handle this situation
id.fastrack <-
  if_else(insurance.data$charges > quantile(insurance.data$charges, 0.95),
          "look",
          "pay") %>% as.factor()
table(id.fastrack)
prop.table(table(id.fastrack))


# plot quantile costs
plot.data <- insurance.data
plot.data$cut <-
  cut(
    plot.data$charges,
    breaks = quantile(
      plot.data$charges,
      probs = seq(0, 1, by = 0.05),
      na.rm = TRUE
    ),
    include.lowest = TRUE
  )
plot.data.long <- plot.data %>% group_by(cut) %>%
  summarise(costs = sum(charges),
            proportion = costs / sum(plot.data$charges)) %>%
  gather(key, value, costs)
ggplot(plot.data.long , aes(x = cut, y = proportion)) + geom_col(alpha =
                                                                   0.7,
                                                                 aes(fill = key, color = key),
                                                                 position = "dodge") + labs(
                                                                   title = "Quantile Plot",
                                                                   subtitle = "Comparing observed vs predicted values",
                                                                   x = "Quantile (obs. Values)",
                                                                   y = "Sum of Loss"
                                                                 ) + theme_light() + theme(axis.text.x  = element_text(
                                                                   angle = 45,
                                                                   vjust = 0.5,
                                                                   size = 6
                                                                 ))


insurance.data.model <-
  as.data.frame(model.matrix( ~ . - 1, insurance.data))
insurance.data.model$sexfemale <- NULL
#View(insurance.data.model)

#Split Data
n = nrow(insurance.data.model)
set.seed(1951)
trainIndex = sample(1:n, size = round(0.7 * n), replace = FALSE)
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
trainTask <- makeRegrTask(data =  training, target = "charges")
#create a task
testnTask <- makeRegrTask(data =  testing, target = "charges")

#Xgboost
#load xgboost
set.seed(1001)
getParamSet("regr.xgboost")

#make learner with inital parameters
xg_set <- makeLearner("regr.xgboost", predict.type = "response")
xg_set$par.vals <- list(
  objective = "reg:gamma",
  eval_metric = "rmse",
  nrounds = 500,
  early_stopping_rounds = 20,
  print_every_n = 10
  
)

#define parameters for tuning
xg_ps <- makeParamSet(
  #makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth", lower = 3, upper = 20),
  makeNumericParam("lambda", lower = 0.55, upper = 0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight", lower = 1, upper = 5),
  makeNumericParam("colsample_bytree", lower = 0.2, upper = 0.8)
)

#define search function
rancontrol <- makeTuneControlRandom(maxit = 5L) #do 100 iterations

#5 fold cross validation
set_cv <- makeResampleDesc("CV", iters = 2L)

#tune parameters
xg_tune <- tuneParams(
  learner = xg_set,
  task = trainTask,
  resampling = set_cv,
  measures = rmse,
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

#set parameters
xg_new <- setHyperPars(learner = xg_set, par.vals = xg_tune$x)

#train model with the best parameters
xgmodel <- train(xg_new, trainTask)

#predict on test data
predict.xg <- predict(xgmodel, testnTask)

str(predict.xg$data)
ggplot(data = predict.xg$data, aes(x = truth, y = response)) + geom_point()


# lares::mplot_lineal(tag = testing$charges,
#                    score = testing$predictions,
#                    subtitle = "Insurance Costs",
#                    model_name = "gbm_model")
#
# lares::mplot_cuts_error(tag = testing$charges,
#                     score = testing$predictions,
#                     title = "Insurance Costs",
#                     model_name = "gbm_model")
#
# lares::mplot_density(tag = testing$charges,
#                      score = testing$predictions,
#                      subtitle = "Insurance Costs",
#                      model_name = "gbm_model")

# lares::mplot_full(tag = predict.xg$data$truth,
#                   score = predict.xg$data$response,
#                   splits = 10,
#                   subtitle = "Insurance Costs",
#                   model_name = "xgb_model",
#                   save = F)

#lares::mplot_splits(tag = testing$charges,
#                    score = testing$predictions,
#                    split = 8)


#--------------    Using Lime to explain Predictions ----------------

library(lime)
test.data <- testing
row.names(test.data) <- id_testing # Add ID
names(test.data)
test.data[, ncol(test.data)] <- NULL

explainer <- lime(
  test.data ,
  xgmodel,
  bin_continuous = TRUE,
  n_bins = 5,
  quantile_bins = FALSE
)

save(explainer, file = "./models/explainer.rda")

explanation <- explain(
  test.data[2, ],
  explainer,
  #n_labels = 1,
  n_features = 6,
  kernel_width = 0.5,
  feature_select = "highest_weights"
)
explanation[, 1:6]

plot_features(explanation, ncol = 2)
plot_explanations(explanation)


#--------------    Using DALEX to explain Predictions ----------------

#https://rawgit.com/pbiecek/DALEX_docs/master/vignettes/DALEX_caret.html#3_classification_use_case_-_wine_data


library(DALEX)

#"Help function for MLR predictions"
custom_predict <- function(object, newdata) {
  pred <- predict(object, newdata = newdata)
  response <- pred$data$response
  return(response)
}

explainer_gbm <- DALEX::explain(
  xgmodel,
  label = "xgmodel",
  data = testing,
  y = testing$charges,
  predict_function = custom_predict
)

mp_gbm <- model_performance(explainer_gbm)

#Plot
plot(mp_gbm)
plot(mp_gbm, geom = "boxplot")

# Variables Importance
vi_classif_gbm <- variable_importance(explainer_gbm)
plot(vi_classif_gbm)

# Partial Dependence plot
pdp_classif_gbm  <-
  variable_response(explainer_gbm, variable = "bmi", type = "pdp")
p1 <- plot(pdp_classif_gbm)
class(p1)

names(testing)
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
  prediction_breakdown(explainer_gbm, observation = testing[2, -ncol(testing)],
                       direction = "down")
plot(observation_explain)

pred <- predict(xgmodel, testnTask)
pred$data[2, ]



# Live package -------------------------------

library(live)
library(mlr)
similar <- sample_locally(
  data = testing,
  explained_instance = testing[2, ],
  explained_var = "charges",
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

plot(trained,  type = "waterfallplot")
plot(trained,  type = "forestplot")


#---------------------- BreakDown package ----------------------------
#https://pbiecek.github.io/breakDown/articles/break_caret.html
#-----------------------------------------------------------------------

library(breakDown)

predict.fun <- function(model, x)
  predict(model, x, type = "raw")
observation_explain <-
  broken(
    xgmodel,
    testing[2, ],
    data = training,
    predict.function = custom_predict,
    direction = "down",
    keep_distributions = TRUE
  )
observation_explain

plot(observation_explain) + ggtitle("breakDown plot for caret/GBM model")



# ------------------------  ceteris Paribus --------------------------

#https://pbiecek.github.io/ceterisParibus/articles/coral.html
#https://pbiecek.github.io/ceterisParibus/articles/ceteris_paribus.html
library("DALEX")
library("ceterisParibus")

explainer_gbm <- DALEX::explain(
  xgmodel,
  label = "xgb",
  data = testing[, -ncol(testing)],
  y = testing$charges,
  predict_function = custom_predict
)


xgmodel$features
names(testing[, -ncol(testing)])
#https://pbiecek.github.io/ceterisParibus/articles/ceteris_paribus.html#cheatsheet
#file:///D:/Users/f003985/Downloads/CeterisParibusCheatsheet.pdf

unlist(testing[2, 9])[1]
cr_rf  <- ceteris_paribus(explainer_gbm, testing[2, -ncol(testing)])

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
                         testing[, -ncol(testing)], y = testing$charges)

plot(
  cp_rf,
  selected_variables = c("age", "bmi"),
  aggregate_profiles = mean,
  show_observations = FALSE
)


# ------------------------ IML ---------------------------------

library("iml")

X = testing[which(names(testing) != "charges")]
predictor = Predictor$new(xgmodel, data = X, y = testing$charges)

#Feature importance
imp = FeatureImp$new(predictor, loss = "mae")
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
interact = Interaction$new(predictor, feature = "children")
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
