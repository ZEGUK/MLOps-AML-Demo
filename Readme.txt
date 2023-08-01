*************************************************************************************
THIS DOCUMENT is written for making the folder organization easy to follow.
*************************************************************************************
Org:

1. MLOps4poc_retraining.ipynb assisted by:
    (1) training.py
    (2) evaluation.py
    (3) register_model.py

    (4) datafiles/
            training_data
            evaluating_data

2. MLOps4poc_deployment_&_call.ipynb assisted by:
    (1) prediction.py

    (2) Postman or python sdk to call webservice

3. azureml-models/:
    created after registering models

4. conda_dependience.yml:
    can be used to deploy models on AML portal

*******************************************************************************************************************************
5. Datastores:<YOUR_DATASTORE_NAME>
    (1) training_dataset
    (2) evaluating_dataset
    (3) output  
*******************************************************************************************************************************
NOTE:
1. Please follow markdowns and guidance comments in both .ipynb and .py scripts.Run the ipynb file CELL BY CELL while take care the comments in it.
2. Please note the source directory whether correct or not. It is recommended to use absolute path instead of relative path.
3. You can use juypter via AML Compute/select compute/juypter or vs code as you preferred.
3. Make sure that the compute instance or cluster used is successfully created ahead.
4. Models are stored in AML temporary data reference and could be viewed in Datastores/pocfoxconn/output by using trained_data = OutputFileDatasetConfig(destination=(datastore, 'output/trained_model')), for example.