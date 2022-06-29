import shutil
from pathlib import Path

import lightning as L
from lightning.app.components.python import TracerPythonScript


class TrainAndSaveModel(TracerPythonScript):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create virtual paths
        self.model_path = "lit://my_trained_model.pt"
        self.log_path = "lit://log.txt"

    def on_after_run(self, _):
        # copy files from local file system to virtual path
        shutil.copy(Path.cwd() / "my_trained_model.pt", self.model_path)
        shutil.copy(Path.cwd() / "log.txt", self.log_path)


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.train_model = TrainAndSaveModel(
            script_path="my_train_script.py",
            script_args=[
                "--num_epochs=1",
                "--model_out=my_trained_model.pt",
                "--log_out=log.txt",
            ],
            cloud_compute=L.CloudCompute("gpu", idle_timeout=60),
        )

    def run(self):
        self.train_model.run()


app = L.LightningApp(RootFlow())
