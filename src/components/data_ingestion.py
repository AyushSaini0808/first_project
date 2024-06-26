import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTranformationConfig
'''This is used for passing inputs to the data ingestion components
   that informs the data ingestion component to know where to store the paths of the 
   train,test and raw data  
'''
@dataclass # allows to directly declare variables and their datatypes in a class
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self) :
        logging.info("Entered the data ingestion method or component.")
        try :
            df=pd.read_csv("notebook/stud.csv")
            logging.info("Reading the CSV file as a DataFrame")
            '''making the directory name according to the specified path and if it already exist 
               let it be. '''
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=23)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data completed. ")
            
            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_transformation(train_data,test_data)