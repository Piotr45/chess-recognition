import os
import shutil
import kaggle


class Data:
    def __init__(self, path):
        self.dataset_path = path
        self.labels = []
        self.get_labels()

    def download_data(self) -> None:
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('niteshfre/chessman-image-dataset', path=self.dataset_path, unzip=True)

        source_dir = f'{self.dataset_path}/Chessman-image-dataset/Chess'

        for file_name in os.listdir(source_dir):
            shutil.move(os.path.join(source_dir, file_name), self.dataset_path)

        shutil.rmtree(f'{self.dataset_path}/Chessman-image-dataset')

    def get_labels(self) -> None:
        self.labels = []
        for label in os.listdir(self.dataset_path):
            self.labels.append(label)

    def create_directories(self, validate: bool = False) -> None:
        os.makedirs(os.path.join(self.dataset_path, "train"))
        os.makedirs(os.path.join(self.dataset_path, "test"))
        if validate:
            os.makedirs(os.path.join(self.dataset_path, "valid"))

    def split_single_data(self, name: str, validate: bool = False) -> None:
        src_directory = f"{self.dataset_path}/{name}"
        file_names = os.listdir(src_directory)

        os.makedirs(f"{self.dataset_path}/train/{name}")
        dest_directory = f"{self.dataset_path}/train/{name}"
        train = file_names[:int(len(file_names) * 0.7)]

        for i, file_name in enumerate(train):
            shutil.move(os.path.join(src_directory, file_name),
                        os.path.join(dest_directory, f"{i:04d}.jpg"))
        if validate:
            tmp = (len(file_names) - len(train)) // 2
            valid = file_names[len(train):len(file_names) - tmp]
            test = file_names[len(file_names) - tmp:]

            os.makedirs(f"{self.dataset_path}/valid/{name}")
            dest_directory = f"{self.dataset_path}/valid/{name}"

            for i, file_name in enumerate(valid):
                shutil.move(os.path.join(src_directory, file_name),
                            os.path.join(dest_directory, f"{i:04d}.jpg"))
        else:
            test = file_names[len(train):]

        os.makedirs(f"{self.dataset_path}/test/{name}")
        dest_directory = f"{self.dataset_path}/test/{name}"

        for i, file_name in enumerate(test):
            shutil.move(os.path.join(src_directory, file_name),
                        os.path.join(dest_directory, f"{i:04d}.jpg"))

        os.rmdir(src_directory)

    def rearrange_directories(self, validate: bool = False) -> None:
        self.create_directories()
        for label in self.labels:
            self.split_single_data(label, validate=validate)
