import lightning as L
from lightning.app.components.python import TracerPythonScript


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.train_model = TracerPythonScript(
            script_path="my_train_script.py",
            script_args=["--num_epochs=1"],
            cloud_compute=L.CloudCompute("gpu", idle_timeout=60),
        )

    def run(self):
        self.train_model.run()


app = L.LightningApp(RootFlow())
