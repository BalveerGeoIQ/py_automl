"""
py_automl
~~~~~~

The py_automl package - a Python package project that is intended
to be used for creating models using geoiq automl platform.
"""

import os
import pandas as pd
import requests
import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
load_dotenv()


headers = {
    'Accept': 'application/json',
    'x-api-key': os.getenv('PROJECT_API_KEY')
}

class automl:
    """ """
    def __init__(self):
        now = datetime.datetime.now()
        print(f"AutoML object is created at {now}")

##### EDA API

    def eda(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :
            

        Returns
        -------

        """
        params = {
            'dataset_id': dataset_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/column/v1.0/getcolumnstatscategorydata', params=params, headers=headers)
        df_nested_list = pd.json_normalize(response.json())
        return pd.DataFrame(df_nested_list['data.geoiq_vars'][0])

#### Datasetinfo

    def dataset_info(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :
            

        Returns
        -------

        """
        params = {
            'dataset_id': dataset_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/dataset/v1.0/getdatasetinfo', params=params, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data'])
        return df_nested_list


#### Chart API

    def variable_dist(self,dataset_id,col_id):
        """

        Parameters
        ----------
        dataset_id :
            
        col_id :
            

        Returns
        -------

        """
        params = {
            'dataset_id': dataset_id,
            'column_id': col_id,
            'scale': 'quantize',
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/column/v1.0/getcolumnfreqgraph', params=params, headers=headers)
        return response.json()



#### Model ID

    def model_id(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :
            

        Returns
        -------

        """
        params = {
            'dataset_id': dataset_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/dataset/v1.0/getdatasetmodels', params=params, headers=headers)
        return response.json()['data'][0]['id']




#### Model Details

    def model_summary(self,model_id):
        """

        Parameters
        ----------
        model_id :
            

        Returns
        -------

        """
        params = {
            'model_id': model_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/model/v1.0/getmodeldetailsevaluate', params=params, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data'])
        return df_nested_list



##### Lift Chart

    def lift_chart(self,model_id):
        """

        Parameters
        ----------
        model_id :
            

        Returns
        -------

        """
        params = {
            'model_id': model_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/model/v1.0/modelliftchartdata', params=params, headers=headers)
        
        
        df_nested_list = pd.DataFrame(response.json()['data'])
        plt.title('Lift chart')
        plt.plot(df_nested_list['x'],df_nested_list['holdout_decile'],label="holdout_decile")
        plt.plot(df_nested_list['x'],df_nested_list['train_decile'],label="train_decile")
        plt.ylabel('Average target value')
        plt.xlabel('Bins based on predicted value')
        plt.legend(loc=4)
        
    

##### ROC Chart

    def roc_chart(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :
            

        Returns
        -------

        """
        
        params = {
            'dataset_id': dataset_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/dataset/v1.0/getdatasetmodels', params=params, headers=headers)
        df_nested_list1 = pd.json_normalize(response.json()['data'])
        
        params = {
            'model_id': df_nested_list1['id'][0],
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/model/v1.0/modelroccurvegraph', params=params, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data'])



        plt.title('ROC Curve')
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_train'],label="Train AUC="+str(df_nested_list1['train_auc'][0]))
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_holdout'],label="Holdout AUC="+str(df_nested_list1['holdout_auc'][0]))
        plt.ylabel('True positive rate (Sensitivity)')
        plt.xlabel('False positive rate (Fallout)')
        plt.legend(loc=4)


##### Feature Importance

    def feature_imp(self,model_id):
        """

        Parameters
        ----------
        model_id :
            

        Returns
        -------

        """
        params = {
            'model_id': model_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/model/v1.0/modelfeatureimp', params=params, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data'])
        return df_nested_list


##### Gain Table

    def holdout_gain_table(self,model_id):
        """

        Parameters
        ----------
        model_id :
            

        Returns
        -------

        """
        json_data = {
            'model_id': model_id,
        }
        response = requests.post('https://app.geoiq.io/cloudmlapis/model/v1.0/getgainstable', json=json_data, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data']['holdout']['data'])
        return df_nested_list
    
    def train_gain_table(self,model_id):
        """

        Parameters
        ----------
        model_id :
            

        Returns
        -------

        """
        json_data = {
            'model_id': model_id,
        }
        response = requests.post('https://app.geoiq.io/cloudmlapis/model/v1.0/getgainstable', json=json_data, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data']['train']['data'])
        return df_nested_list
    



##### Confusion Matrix

    def confusion_matrix(self,model_id,dataset_id,threshold):
        """

        Parameters
        ----------
        model_id :
            
        dataset_id :
            
        threshold :
            

        Returns
        -------

        """
        json_data = {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'threshold': str(threshold),
        }
        response = requests.post('https://app.geoiq.io/cloudmlapis/model/v1.0/modelconfusionmatrix', json=json_data, headers=headers)
        return response.json()



##### ModelDetails

    def model_details(self,model_id):
        """

        Parameters
        ----------
        model_id :
            

        Returns
        -------

        """
        params = {
            'model_id': model_id,
        }
        response = requests.get('https://app.geoiq.io/cloudmlapis/model/v1.0/getmodelproperties', params=params, headers=headers)
        df_nested_list = pd.json_normalize(response.json()['data'])
        return df_nested_list

