import numpy as np
import pandas as pd
import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

from sklearn.metrics import  r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from xgboost import XGBRegressor

import warnings

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Initiating model training.")
            logging.info("Split training and testing data.")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]                
            )
            models={
                "K Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "XGB Regressor":XGBRegressor(),
                "Linear Regression":LinearRegression(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Ridge Regressor":Ridge(),
                "Lasso Regressor":Lasso(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,y_test=y_test,X_test=X_test,models=models)
            
            # To get best model score
            best_model_score=max(sorted(model_report.values()))
            # To get best model
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]    
            
            if best_model_score<0.6:
                raise CustomException("No best model found !")
            logging.info("Best model found for both training and testing data.")
            save_object(file_path=self.trainer_config.trained_model_file_path,
                obj=best_model)
            predictions=best_model.predict(X_test)
            rsquare=r2_score(y_test,predictions)
            return rsquare
        except Exception as e:
            raise CustomException(e,sys)
        