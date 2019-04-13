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
summary(insurance.data)
names(insurance.data)  <- make.names(names(insurance.data))
View(insurance.data)
str(insurance.data)
summarizeColumns(insurance.data)
id<-insurance.data$competitorname
names(insurance.data)


table(insurance.data$region)

# Basic density
ggplot(insurance.data, aes(x=charges, y=..density.., color = sex, fill = sex), legend=TRUE) +
  labs(title="Charge amount") +
  geom_density(alpha = 0.2)  + geom_vline(data=insurance.data, aes(xintercept=mean(charges), colour=sex), linetype="dashed", size=0.5)  + theme_minimal()

id.fastrack <- if_else(insurance.data$charges > quantile(insurance.data$charges,0.95),"look","pay") %>% as.factor()
table(id.fastrack)
prop.table(table(id.fastrack))

# plot quantile costs
plot.data <- insurance.data
plot.data$cut <- cut(plot.data$charges,breaks=quantile(plot.data$charges, probs=seq(0,1, by=0.05), na.rm=TRUE),include.lowest=TRUE)
plot.data.long<- plot.data %>% group_by(cut) %>% 
  summarise(costs=sum(charges), proportion=costs/sum(plot.data$charges)) %>%  
  gather(key,value, costs) 

ggplot(plot.data.long , aes(x=cut,y=proportion)) + geom_col(alpha=0.7,aes(fill = key,color=key),position = "dodge") + labs(title = "Quantile Plot", subtitle = "Comparing observed vs predicted values", x = "Quantile (obs. Values)", y="Sum of Loss")+ theme_light() +theme(axis.text.x  = element_text(angle=45, vjust=0.5, size=6)) 

insurance.data.model<- as.data.frame(model.matrix(~ . -1, insurance.data))
insurance.data.model$sexfemale<-NULL

n = nrow(insurance.data.model)
set.seed(1951)
trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
id <- 1:n
training = insurance.data.model[trainIndex ,]
id_training <- id[trainIndex]
testing = insurance.data.model[-trainIndex ,]
id_testing <- id[-trainIndex]

lapply(insurance.data.model,class)
#View(insurance.data.model)

View(insurance.data.model)
names(insurance.data.model)

names(training)
#-------------------------   Prep. Experience -----------------------

fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=3,
                           allowParallel=T)


## -------------------------------  GBM ------------------------------

grid <- expand.grid(interaction.depth=c(2,4,6), # Depth of variable interactions
                     n.trees=c(250,500),	        # Num trees to fit
                     shrinkage=c(0.01,0.1),		# Try 2 values for learning rate 
                     n.minobsinnode = c(10,20))	# min obs in node to be able to slip

#grid <- expand.grid(mtry=c(1,2,3,4,5,6))

set.seed(1951)
names(training)
class(training)

#training <- training[order(training$bmi),]
set.seed(1234)
#test <- training %>% select(-winpercent)
gbmFit1 <- caret::train(charges~.,
                 data=training,
                 method = "gbm", 
                 #rounds=50,
                 trControl = fitControl,
                 tuneGrid=grid,
                 metric="RMSE",
                 verbose = FALSE)

gbmFit1
plot(gbmFit1)

plot(varImp(gbmFit1))
#test predictions
test.pred <- predict(gbmFit1,testing)
test1<-test.pred
test2<-test.pred
View(test1)
1-sum(test1)/sum(test2)
View(test2)
testing$predictions<-test.pred
#View(testing)
save(gbmFit1, file = "./models/bgmModel.rda")


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

lares::mplot_full(tag = testing$charges, 
                  score = testing$predictions,
                  splits = 10,
                  subtitle = "Insurance Costs",
                  model_name = "gbm_model",
                  save = F)

#lares::mplot_splits(tag = testing$charges, 
#                    score = testing$predictions,
#                    split = 8)


#Using Lime to explain Predictions
library(lime)
row.names(testing) <- id_testing # Add ID
names(testing)
testing[,ncol(testing)] <- NULL
explainer <- lime(testing[,-ncol(testing)] , gbmFit1, 
                  bin_continuous = TRUE,
                  n_bins = 5, 
                  quantile_bins = FALSE)

save(explainer, file = "./models/explainer.rda")

class(testing[2,])
explanation <- explain(testing[2,-ncol(testing)], 
                       explainer, 
                       #n_labels = 1,
                       n_features = 4,
                       kernel_width = 0.5,
                       feature_select="highest_weights")
explanation[, 1:9]

plot_features(explanation, ncol = 2)
plot_explanations(explanation)


#https://rawgit.com/pbiecek/DALEX_docs/master/vignettes/DALEX_caret.html#3_classification_use_case_-_wine_data


library(DALEX)
explainer_gbm <- DALEX::explain(gbmFit1, label="gbm", 
                                    data = testing, y =testing$charges)

mp_gbm <- model_performance(explainer_gbm)

plot(mp_gbm)
plot(mp_gbm, geom = "boxplot")

# Variables Importance
vi_classif_gbm <- variable_importance(explainer_gbm)
plot(vi_classif_gbm)

# Partial Dependance plot
pdp_classif_gbm  <- variable_response(explainer_gbm, variable = "bmi", type = "pdp")
plot(pdp_classif_gbm)

# Acumulated Local Effects plot
ale_classif_gbm <- variable_response(explainer_gbm, variable = "age", type = "ale")
plot(ale_classif_gbm)

#explain Observation
observation_explain <- prediction_breakdown(explainer_gbm,observation = testing[2,-ncol(testing)],direction = "up",)
testing[2,]
plot(observation_explain)

predict(gbmFit1,testing[2,-ncol(testing)])

# Live package -------------------------------

library(live)
library(mlr)
similar <- sample_locally(data = testing,
                          explained_instance = testing[2,],
                          explained_var = "charges",
                          size = 20)

similar1 <- add_predictions2(to_explain = similar,
                             black_box_model = gbmFit1)

trained <- fit_explanation(live_object = similar1,white_box = "regr.lm",selection = FALSE)


plot_explanation(trained, "waterfallplot", explained_instance = testing[2,])
plot_explanation(trained, "forestplot", explained_instance = testing[2,])


#---------------------- BreakDown package ----------------------------
#https://pbiecek.github.io/breakDown/articles/break_caret.html

#-----------------------------------------------------------------------

library(breakDown)
predict.fun <- function(model, x) predict(model, x, type = "raw")
observation_explain <- broken(gbmFit1, testing[2,-61], data = training, predict.function = predict.fun)
observation_explain

plot(observation_explain) + ggtitle("breakDown plot for caret/GBM model")



# ------------------------  ceteris Paribus --------------------------

#https://pbiecek.github.io/ceterisParibus/articles/coral.html
#https://pbiecek.github.io/ceterisParibus/articles/ceteris_paribus.html
library("DALEX")
library("ceterisParibus")

explainer_gbm <- DALEX::explain(gbmFit1$finalModel, label="gbm", 
                                data = testing[,-ncol(training)], y =testing$charges,n.trees=gbmFit1$finalModel$n.trees)

#https://pbiecek.github.io/ceterisParibus/articles/ceteris_paribus.html#cheatsheet

unlist(testing[2,9])[1]
cr_rf  <- ceteris_paribus(explainer_gbm, testing[2,-9])

plot(cr_rf, plot_residuals = FALSE,selected_variables = c("age","bmi","children"))
plot(cr_rf, plot_residuals = TRUE,selected_variables = c("age","bmi","children"))


neighbours <- select_neighbours(testing, testing[2,], n = 15)
cr_rf  <- ceteris_paribus(explainer_gbm, neighbours)
plot(cr_rf,show_profiles = TRUE, show_observations = TRUE, show_rugs = TRUE, plot_residuals = TRUE,selected_variables = c("age","bmi","children"))


View(training)
# ------------------------ IML ---------------------------------


library("iml")


X = testing[which(names(testing) != "charges")]
predictor = Predictor$new(gbmFit1, data = X, y = testing$charges)

#Feature importance
imp = FeatureImp$new(predictor, loss = "ce")
plot(imp)

#Partial dependence
pdp.obj = Partial$new(predictor, feature = "age")
pdp.obj$plot()


pdp.obj$set.feature("bmi")

pdp.obj$plot()

pdp.obj$center(min(testing$age))
pdp.obj$plot()

#Measure interactions
interact = Interaction$new(predictor)
plot(interact)

#Surrogate model
tree = TreeSurrogate$new(predictor, maxdepth = 4)
plot(tree)


#Explain single predictions with a local model

lime.explain = LocalModel$new(predictor, x.interest = X[1,])
plot(lime.explain)



#Explain single predictions with game theory

#Shapley value
shapley = Shapley$new(predictor, x.interest = X[1,])
shapley$plot()

results = shapley$results
head(results)
