import os
import glob
import src.test
import src.train
from src.data import Data


if __name__ == '__main__':
    data = Data(f"{os.getcwd()}/data")
    data.rearrange_directories(validate=True)
