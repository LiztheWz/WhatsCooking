# library(jsonlite)
# library(tidytext)
# library(tidyverse)
# library(stringr)
# library(dplyr)
# library(tidymodels)
# library(vroom)
# > 0.72495


# train_cooking <- read_file("train.json") %>% fromJSON()
# test_cooking <- read_file("test.json") %>% fromJSON()
# train_cooking <- train_cooking %>% unnest(ingredients)
# test_cooking <- test_cooking %>% unnest(ingredients)
# 
# 
# train_cooking <- train_cooking %>%
#   group_by(id, cuisine) %>% 
#   summarize(cuisine = first(cuisine),
#             num_ingredients = n(),
#             has_salt = as.integer(any(ingredients == "salt")),
#             has_milk = as.integer(any(ingredients == "milk")),
#             has_sugar = as.integer(any(ingredients == "sugar")),
#             # ingredients = list(ingredients),
#             .groups = "drop")
# 
# test_cooking <- test_cooking %>%
#   group_by(id) %>% 
#   summarize(num_ingredients = n(),
#             has_salt = as.integer(any(ingredients == "salt")),
#             has_milk = as.integer(any(ingredients == "milk")),
#             has_sugar = as.integer(any(ingredients == "sugar")),
#             # ingredients = list(ingredients),
#             .groups = "drop")
# 
# 
# cooking_recipe <- recipe(cuisine ~ ., data = train_cooking) %>%
#   step_mutate(across(where(is.character), as.factor)) %>% 
#   step_dummy(all_nominal_predictors())
# 
# 
# rando_mod <- rand_forest(mtry = tune(),
#                               min_n = tune(),
#                               trees = 500) %>%
#   set_engine("ranger") %>% set_mode(mode = "classification")
# 
# cooking_workflow <- workflow() %>%
#   add_recipe(cooking_recipe) %>%
#   add_model(rando_mod)
# 
# rf_params <- parameters(mtry(),
#                         min_n()) %>% finalize(train_cooking)
# 
# tuning_grid <- grid_regular(rf_params,
#                             levels = 5) # L^2 total tuning possibilities
# 
# folds <- vfold_cv(train_cooking, v = 3, repeats = 1)
# 
# CV_results <- cooking_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# # roc_auc, f_meas, sens, recall, spec, precision, accuracy
# 
# bestTune <- CV_results %>% select_best(metric = "roc_auc")
# 
# final_wf <- cooking_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train_cooking)
# 
# cooking_predictions <- final_wf %>%
#   predict(new_data = test_cooking, type = "class") %>%
#   bind_cols(test_cooking %>% select(id)) %>%
#   select(id, cuisine = .pred_class)
# 
# vroom_write(cooking_predictions, "random_forest_predictions.csv", delim = ',')



# TFIDF
# ============================================================================
library(jsonlite) 
library(tidytext)
library(tidyverse)
library(stringr)
library(dplyr)
library(tidymodels)
library(vroom)
library(textrecipes)

train_cooking <- read_file("train.json") %>% fromJSON()
test_cooking <- read_file("test.json") %>% fromJSON()

tfidf_recipe <- recipe(cuisine ~ ingredients, data = train_cooking) %>%
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens = 500) %>%
  step_tfidf(ingredients)


rando_mod <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 1000) %>%
  set_engine("ranger") %>% set_mode(mode = "classification")

cooking_workflow <- workflow() %>%
  add_recipe(tfidf_recipe) %>%
  add_model(rando_mod)

rf_params <- parameters(mtry(),
                        min_n()) %>% finalize(train_cooking)

tuning_grid <- grid_regular(rf_params,
                            levels = 5) # L^2 total tuning possibilities

folds <- vfold_cv(train_cooking, v = 3, repeats = 1)

CV_results <- cooking_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))
# roc_auc, f_meas, sens, recall, spec, precision, accuracy

bestTune <- CV_results %>% select_best(metric = "roc_auc")

final_wf <- cooking_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_cooking)

cooking_predictions <- final_wf %>%
  predict(new_data = test_cooking, type = "class") %>%
  bind_cols(test_cooking %>% select(id)) %>%
  select(id, cuisine = .pred_class)

vroom_write(cooking_predictions, "tfidf_predictions3.csv", delim = ',')




# DATA ROBOT
# ============================================================================
train_cooking <- read_file("train.json") %>% fromJSON()
train_cooking <- train_cooking %>% 
  mutate(ingredients_text = sapply(ingredients, paste, collapse = ", ")) %>%
  select(-ingredients)

test_cooking <- read_file("test.json") %>% fromJSON()
test_cooking <- test_cooking %>% 
  mutate(ingredients_text = sapply(ingredients, paste, collapse = ", ")) %>%
  select(-ingredients)


vroom_write(train_cooking, "training_cooking.csv", delim = ',')
vroom_write(test_cooking, "test_cooking.csv", delim = ',')

dr_train <- vroom("dr_nn.csv")
dr_train <- dr_train %>%
  mutate(cuisine = cuisine_PREDICTION) %>%
  select(id, cuisine)

vroom_write(dr_train, "dr_nn_predictions.csv", delim = ',')


