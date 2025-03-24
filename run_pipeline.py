from pipelines.training_pipelines import train_pipeline

if __name__ == "__main__":
    train_pipeline(
        data_path="data/olist_customers_dataset.csv"
    )