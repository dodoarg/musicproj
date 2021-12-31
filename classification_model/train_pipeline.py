from processing.data_manager import save_pipeline
from training import train_pipeline_on_training_data

if __name__ == "__main__":
    popularity_pipe = train_pipeline_on_training_data()
    save_pipeline(pipeline_to_persist=popularity_pipe)
