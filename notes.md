#### Steps:

1. Download Dataset from : https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download and put that file in data folder.
   1. The review_score is the target variable
   2. Rest are features
2. Create a virtual env using conda :
   1. Copied over requirements.txt from git
   2. create conda environment :
      1. ```shell
         conda create -n zenml-customer-satisfaction python=3.11
         conda activate zenml-customer-satisfaction
         pip install "zenml["server"]"
         ```
      2. ```shell
         zenml init
         zenml login --local
         ```
      3. If there are any warnings, run 'zenml downgrade'
3. Create folders:
   1. src - This will contain all files related to model training
   2. pipelines - This will contain all files related to pipelines
   3. saved_model - Where we store are saved models
   4. steps - All the components that the tasks need to be done
4. Create files:
   1. __init__.py
   2. run_pipeline.py
   3. Create blueprint of steps
      1. steps/ingest_data.py - load data for model
      2. steps/clean_data.py - cleaning the data
      3. steps/train_model.py - for training the model
      4. steps/evaluation.py - to evaluate
   4. Create Pipeline :
      1. pipelines/training_pipeline.py
      2. run_pipeline.py which will call the training pipeline `run_pipeline.py`
5. After Skeleton Code is completed -
   1. src/data_cleaning.py - Writing the Strategy Pattern for cleaning and splitting the data into testing and training. Update the clean_data function in steps to do the actual cleaning.
   2.
