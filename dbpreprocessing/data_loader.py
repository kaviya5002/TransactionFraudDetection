import pandas as pd
import os

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        """Loads the dataset from the given path."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        data = pd.read_csv(self.dataset_path)
        return data

    def summarize_data(self, data):
        """Prints basic information about the dataset."""
        print("🔹 Shape of the dataset:", data.shape)
        print("\n🔹 Columns:", data.columns.tolist())
        print("\n🔹 Missing values:\n", data.isnull().sum().head())
        print("\n🔹 Class distribution:\n", data['Class'].value_counts())
