import os
import glob
from src.model import ModelHandler
from src.datahandler import DataHandler

if __name__ == '__main__':
    dataset_path = f"{os.getcwd()}/data"
    data = DataHandler(dataset_path)
    # data.download_data()
    # data.rearrange_directories(validate=True)
    model_handler = ModelHandler(dataset_path)
    model_handler.show_model_summary()
    model_handler.show_accuracy_plots()
    model_handler.evaluate()
    # model_handler.save_model()
    model_handler.plot_predict()
    model_handler.show_conf_matrix()

