import os
from pathlib import Path
import pandas as pd

class DataIngestion:
    CONTAINER_INPUT  = "/opt/ml/processing/input"
    CONTAINER_OUTPUT = "/opt/ml/processing/ingested"
    LOCAL_INPUT = "/home/ec2-user/SageMaker/data"
    LOCAL_OUTPUT = "/home/ec2-user/SageMaker/ingested"
    FILENAME = "data_D.csv"

    def __init__(self):
        in_container = os.path.exists(self.CONTAINER_INPUT)
        self.input_dir  = self.CONTAINER_INPUT  if in_container else self.LOCAL_INPUT
        self.output_dir = self.CONTAINER_OUTPUT if in_container else self.LOCAL_OUTPUT
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> str:
        input_file = os.path.join(self.input_dir, self.FILENAME)
        output_file = os.path.join(self.output_dir, self.FILENAME)

        if not os.path.exists(input_file):
            available = os.listdir(self.input_dir) if os.path.exists(self.input_dir) else "directory not found"
            raise FileNotFoundError(f"{input_file} tidak ditemukan")

        df = pd.read_csv(input_file)
        assert not df.empty, "Dataset kosong"

        df.to_csv(output_file, index=False)
        print(f"Selesai")
        return output_file

if __name__ == "__main__":
    DataIngestion().run()
