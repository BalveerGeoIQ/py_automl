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
# from dotenv import load_dotenv
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
import time
import ast
import json
# load_dotenv()


# self.headers = {
#   'x-api-key': os.getenv('PROJECT_API_KEY'),
#   'Content-Type': 'application/json'
# }
#     os.getenv('PROJECT_API_KEY')


class automl:
    """ """
    def __init__(self, authorization_key):
        self.headers = {
            'x-api-key': authorization_key,
            'Content-Type': 'application/json'
        }
        now = datetime.datetime.now()
        print(f"AutoML object is created at {now}")
        
##### Data Upload API

    def create_dataset(self,df, dataset_name ='test' ,dv_col = 'dv', dv_positive = '1',latitude_col = "geo_latitude" ,
          longitude_col = "geo_longitude",unique_col = 'geoiq_identifier_col',geocoding = 'false',
          address_col = '\'\'', pincode_col = '\'\'' , additional_vars = '[]' ):
        """

        Parameters
        ----------
        df : A dataframe containing the user data, which should contains target variable, customer location details, and additional variables. 
            
        dataset_name : Name of the dataset
             (Default value = 'test')
        dv_col : Target variable column name
             (Default value = 'dv')
        dv_positive : Specify the good observation in the dataset 
             (Default value = '1')
        latitude_col : Latitutde column in the dataframe
             (Default value = "geo_latitude")
        longitude_col : Latitutde column in the dataframe
             (Default value = "geo_longitude")
        unique_col : Unique identifier in the dataset
             (Default value = 'geoiq_identifier_col')
        geocoding : If geocoding is required need to be true
             (Default value = 'false')
        address_col : Address column in the dataframe
             (Default value = '\'\'')
        pincode_col : Pincode column in the dataframe
             (Default value = '\'\'')
        additional_vars : Additional variables need to be passed in the creation of the model
             (Default value = '[]')

        Returns
        -------

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/dataset/v1.0/createdataset"

        df.to_csv(f'/tmp/{dataset_name}.csv',index=False)
        
#         self.headers = {
#           'x-api-key': s
#         }

        payload={'dataset_name': dataset_name,
        'dv_col': dv_col,
        'dv_positive': dv_positive,
        'latitude':latitude_col,
        'longitude':longitude_col,
        'unique_col': unique_col,
        'geocoding': geocoding,
        'address': address_col,
        'pincode': pincode_col,
        'user_selected_vars': additional_vars}
        
        files=[
          ('df_data',(f'{dataset_name}.csv',open(f'/tmp/{dataset_name}.csv','rb'),'text/csv'))
        ]
        headers = {'x-api-key' : self.headers['x-api-key']}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
    
        os.remove(f'/tmp/{dataset_name}.csv')
        
        print(response.text)
        dataset_id = response.json()['data']['dataset_id']
                
        return dataset_id
    
    
##### Data Enrichment


    def data_enrichment(self,dataset_id, var_list):
        """

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        var_list : List of variables need to be extracted
            

        Returns
        -------
        A url through which enriched data conatining geoiq variables can be downloaded. 
        

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/dataset/v1.0/downloadgeoiqvarsdata"

        payload = json.dumps({
          "dataset_id": dataset_id,
          "var_list": str(var_list)
        })
        self.headers = {
          'x-api-key': 'geoiq@1234',
          'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=self.headers, data=payload)

        print(response.json()['download_url'])

        
        
#### Describe Dataset


    def describe_dataset(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            

        Returns
        -------
        Return a dict containing the dataset creation progress status details

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/progress/v1.0/getdatasetprogress"
        payload = json.dumps({
          "dataset_id": dataset_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        
        if (response.json()['data']['progress'][0]['status_text']) == 'complete':
            url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getalldatasetmodels"

            payload = json.dumps({
              "dataset_id": dataset_id
            })

            response = requests.request("GET", url, headers=self.headers, data=payload)
            print(f"Dataset creation is comleted for this dataset id {dataset_id}")
            return response.json()['data'][0]['model_id']
        else:
            return (response.json()['data']['progress'][0])
            
    

##### Progress Bar API

    def progress_status(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :
            

        Returns
        -------

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/progress/v1.0/getdatasetprogress"
        payload = json.dumps({
          "dataset_id": dataset_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)

        return (response.text)
    
    def make_clickable(self,val):
        """

        Parameters
        ----------
        val :
            

        Returns
        -------

        """
        # target _blank to open new window
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)
    



    
    
## MODEL GET ALL DATASET


    def list_datasets(self):
        """

        Parameters
        ----------
        No Parameters required
            

        Returns
        -------
        Lists alll the datasets created by the user
        

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/dataset/v1.0/getalldataset"

        payload={}

        response = requests.request("POST", url, headers=self.headers, data=payload)
        y = pd.DataFrame(response.json()['data'])
        
        y.style.set_properties(**{'url': self.make_clickable})
        print(y)
        return y
    
## GET ALL DATASET MODELS

    def list_models(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            

        Returns
        -------
        Lists alll the models created on the given dataset

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getalldatasetmodels"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)
        
        y = pd.DataFrame(response.json()['data'])
      
        y.style.set_properties(**{'url': self.make_clickable})
        print(y)
        return y


        
##### EDA API

    def eda(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :  An unique identifier for the dataset
            

        Returns
        -------
        Returns the dataframe containing exploratory data analysis on the given dataframe


        """
#         params = {
#             'dataset_id': dataset_id,
#         }
#         response = requests.get('https://app.geoiq.io/cloudmlapis/column/v1.0/getcolumnstatscategorydata', params=params, headers=self.headers)
        url = "https://app.staging.geoiq.io/automlwrapper/stg/dataset/v1.0/getdataseteda"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
    
        df_nested_list = pd.json_normalize(response.json())

        return pd.DataFrame(df_nested_list['data.data.geoiq_vars'][0])[[ 'column_name', 'column_type','iv','auc_1', 'auc_2', 'auc_3', 'bins',
       'catchment', 'category', 'F_test_pvalue', 'T_test_pvalue', 
       'desc_name', 'description', 'deviation', 'direction', 'id', 'ks',
       'major_category', 'max_ks', 'mean', 'name', 'normalization_level',
       'normalization_level_id', 'roc', 'sd', 'sub_category_name', 'unique',
       'unique_count', 'variable', 'vhm_hierarchy_id']].sort_values('iv',ascending=False).reset_index(drop=True)

#### Datasetinfo


    def dataset_info(self,dataset_id):
        """

        Parameters
        ----------
        dataset_id :  An unique identifier for the dataset
            

        Returns
        -------

        """
        
        url = "https://app.staging.geoiq.io/automlwrapper/stg/dataset/v1.0/getdatasetinfo"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        
#         y = pd.DataFrame([response.json()['data']['data']]).T
#         y.columns = ['Dataset_Info']
#         y.loc['created_at','Dataset_Info'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(y.loc['created_at','Dataset_Info']))
#         y.loc['updated_at','Dataset_Info'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(y.loc['updated_at','Dataset_Info']))
        
        response.json()['data']['data']['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response.json()['data']['data']['created_at']))
        response.json()['data']['data']['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response.json()['data']['data']['updated_at']))
    
        return response.json()['data']['data']



#### Chart API


    def variable_distribution_plot(self,dataset_id,variable_name, quantize=True):
        """

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        variable_name : variable to be plotted
            
        quantize : To plot the quantize graph.
             (Default value = True)

        Returns
        -------
        Plot the variable distribution
        

        """
        
        url = "https://app.staging.geoiq.io/automlwrapper/stg/dataset/v1.0/getdataseteda"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
    
        df_nested_list = pd.json_normalize(response.json())
        df_nested_list = pd.DataFrame(df_nested_list['data.data.geoiq_vars'][0])
        
        xlabel = df_nested_list[df_nested_list['column_name']==variable_name]['description'].values[0]
        
        ### Graph API
        
        url = "https://app.staging.geoiq.io/automlwrapper/stg/column/v1.0/getcolumnfrequencygraph"
        
        col_id =  df_nested_list[df_nested_list['column_name']==variable_name]['id'].values[0]

        payload = json.dumps({
          "dataset_id": dataset_id,
          "column_id": col_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        if quantize ==True:
            df_nested_list = pd.json_normalize(response.json()['data']['quantize_data'])
        else:
            df_nested_list = pd.json_normalize(response.json()['data']['quantile_data'])
            
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot()
        if quantize ==True:
            ax1.set_title('EDA quantize chart')
        else:
            ax1.set_title('EDA quantile chart')

        ax2 = ax1.twinx()
        ax1.bar(df_nested_list['range'],df_nested_list['count'],label="Count", color='r')
        ax2.plot(df_nested_list['range'],df_nested_list['avg_dv'],label="% of positive observation", color='b')

        fig.autofmt_xdate(rotation=45)
        # ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Number of rows', color='r')
        ax2.set_ylabel('% of positive observation', color='b')
        fig.legend(loc=1)
        plt.show()


#### Model ID

#     def model_id(self,dataset_id):
#         """

#         Parameters
#         ----------
#         dataset_id :
            

#         Returns
#         -------

#         """
#         params = {
#             'dataset_id': dataset_id,
#         }
#         response = requests.get('https://app.geoiq.io/cloudmlapis/dataset/v1.0/getdatasetmodels', params=params, headers=self.headers)
#         return response.json()['data'][0]['id']

##### Custom Model
    def create_custom_model(self,dataset_id,model_name,model_type = "xgboost", split_ratio ="[0.7,0.3,None]"):
        """

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        model_name : Model name
            
        model_type : Model type ['logistic', 'xgboost']
             (Default value = "xgboost")
        split_ratio : Train-Validation-Test ratio
             (Default value = "[0.7)
        0.3 :
            
        None]" :
            

        Returns
        -------
        Returns the model id
        

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/createmodel"

        payload = json.dumps({
          "dataset_id": dataset_id,
          "model_name": model_name,
          "model_type": model_type,
          "split_ratio": split_ratio
        })
 

        response = requests.request("POST", url, headers=self.headers, data=payload)
        model_id = ast.literal_eval(response.text)['data']['model_id']
        
#         i = 0
#         pbar = tqdm(total=2)
        
#         progress_state = 0
#         pt = 0
#         while int(ast.literal_eval(self.model_progress(model_id))['data']['progress'][0]['progress_stage']) > pt :

#             if int(ast.literal_eval(self.model_progress(model_id))['data']['progress'][0]['progress_stage']) ==100:
#                 pt = 101 
#             if int(ast.literal_eval(self.model_progress(model_id))['data']['progress'][0]['progress_stage']) - progress_state >0 :
#                 pbar.update(1)

#             z = ast.literal_eval(self.model_progress(model_id))
#             progress_state = int(z['data']['progress'][0]['progress_stage'])
#             x = dict(reversed(list(z.items())))

#             s1 = x['data']['progress'][i]['stage']
#             s2 = x['data']['progress'][i]['status_text']
#             pbar.set_description("Stage: %s" % s1)
#             time.sleep(1)
#             pbar.set_postfix({' -----> Status' : s2})

        return model_id
    
        
#### Describe Model


    def describe_model(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        Return a dict containing the model progress status details
        

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/progress/v1.0/getmodelprogress"
        payload = json.dumps({
          "dataset_id": model_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)

        return (response.json()['data']['progress'][0])

        
#### Model Progress


#     def model_progress(self,model_id):
#         """

#         Parameters
#         ----------
#         model_id : An unique identifier for the model
            

#         Returns
#         -------
        

#         """

#         url = "https://app.staging.geoiq.io/automlwrapper/stg/progress/v1.0/getmodelprogress"

#         payload = json.dumps({
#           "model_id": model_id
#         })
        
    
#         response = requests.request("POST", url, headers=self.headers, data=payload)

#         return response.json()['data']['progress'][0]

#### Model Details

    def model_summary(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        A dict containing performance metrics of the model

        """
        
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getmodeloverview"

        payload = json.dumps({
          "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)
        
        y = pd.DataFrame([response.json()['data']]).T
        y.columns = ['Model_Results']
        y.loc['completed_at','Model_Results'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(y.loc['completed_at','Model_Results']))
        y.loc['started_at','Model_Results'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(y.loc['started_at','Model_Results']))

        return y
        


## LIFT Chart

    def create_lift_chart(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        Plots the lift chart

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getmodelliftchart"

        payload = json.dumps({
          "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)

        df_nested_list = pd.DataFrame(response.json()['data'])
        plt.title('Lift chart')
        plt.plot(df_nested_list['x'],df_nested_list['holdout_decile'],label="holdout_decile")
        plt.plot(df_nested_list['x'],df_nested_list['train_decile'],label="train_decile")
        plt.ylabel('Average target value')
        plt.xlabel('Bins based on predicted value')
        plt.legend(loc=4)
        
    

##### ROC Chart

    def create_roc_chart(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        Plot Receiver operating characteristic (ROC) curve
        

        """
        
        
        ## AUC values
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getmodeloverview"

        payload = json.dumps({
          "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)

        df_nested_list1 = pd.json_normalize(response.json()['data'])
        
        ## Roc
        
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getmodelroccurve"


        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.DataFrame(response.json()['data'])

        plt.title('ROC Curve')
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_train'],label="Train AUC="+str(df_nested_list1['train_auc'][0]))
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_holdout'],label="Holdout AUC="+str(df_nested_list1['holdout_auc'][0]))
        plt.ylabel('True positive rate (Sensitivity)')
        plt.xlabel('False positive rate (Fallout)')
        plt.legend(loc=4)


##### Feature Importance

    def get_feature_importance(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        A dataframe containing feature importance
        

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getmodelfeatureimportance"

        payload = json.dumps({
          "model_id": model_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)

        return pd.json_normalize(response.json()['data'])


##### Gain Table

    def get_gains_table(self,model_id,split = 'train'):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            
        split : 
             (Default value = 'train')

        Returns
        -------

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getgainstable"

        payload = json.dumps({
          "model_id": model_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        
        if split == 'train':
            df_nested_list = pd.json_normalize(response.json()['data'][split]['data'])
            df_nested_list.rename(columns={'abs_diff_array_train':'KS', 'bucket_data_array':'Score Range', 'decile_data_array':'Decile',
       'percentage_data_array_train':'Positive Class Percentage', 'target_0_array_train':'Non-positive Class Count',
       'target_array_train':'Positive Class Count'},inplace=True)

            df_nested_list = df_nested_list[['Decile', 'Score Range', 'Non-positive Class Count', 'Positive Class Count','KS','Positive Class Percentage']]
        
        elif split == 'holdout':
            df_nested_list = pd.json_normalize(response.json()['data'][split]['data'])
            df_nested_list.rename(columns={'abs_diff_array_holdout':'KS', 'bucket_data_array':'Score Range', 'decile_data_array':'Decile',
       'percentage_data_array_holdout':'Positive Class Percentage', 'target_0_array_holdout':'Non-positive Class Count',
       'target_array_holdout':'Positive Class Count'},inplace=True)

            df_nested_list = df_nested_list[['Decile', 'Score Range', 'Non-positive Class Count', 'Positive Class Count','KS','Positive Class Percentage']]
        
        else:
            print(" Pass correct value in the split argument")
            
        return df_nested_list



##### Confusion Matrix

    def get_confusion_matrix(self,dataset_id,model_id,threshold):
        """

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        model_id : An unique identifier for the model
            
        threshold :
            

        Returns
        -------

        """
        url = "https://app.staging.geoiq.io/automlwrapper/stg/model/v1.0/getmodelconfusionmatrix"

        payload = json.dumps({
        'model_id': model_id,
        'dataset_id': dataset_id,
        'threshold': str(threshold),
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        return pd.json_normalize(response.json()['data'])



