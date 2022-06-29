from pathlib import Path

import lightning as L
from lightning.app.components.python import TracerPythonScript
from lightning.app.storage import Drive


class TrainAndSaveModel(TracerPythonScript):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.checkoint_files = Drive("lit://my_outputs")

    def on_after_run(self, _):
        self.checkoint_files.put(Path.cwd() / "my_trained_model.pt")
        self.checkoint_files.put(Path.cwd() / "log.txt")


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
