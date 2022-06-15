import cv2
import gradio as gr

# From local directory:
import imgproc
import lightning as L
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from lightning.app.components.serve import ServeGradio
from lightning_app.frontend import StreamlitFrontend
from model import Generator


class SRGAN(ServeGradio):

    inputs = gr.inputs.Image(type="pil", label="Select an input image")  # required
    outputs = gr.outputs.Image(type="pil")  # required
    examples = [["./examples/comic_lr.png"]]  # required

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ready = False  # required

    def predict(self, img):

        height, width = img.size
        print("Original size:", height, width)
        max_size = max(height, width)
        if max_size > 100:
            ratio = 100 / max_size
            new_size = (round(ratio * height), round(ratio * width))
            img = img.resize(new_size)

        new_height, new_width = img.size
        print("Resized size:", new_height, new_width)

        DEVICE = torch.device("cpu")
        # Read LR image and HR image

        # convert PIL to open cv
        opencv_image = np.array(img)
        opencv_image = opencv_image[:, :, ::-1].copy()
        lr_image = opencv_image.astype(np.float32) / 255.0

        # Convert BGR channel image format data to
        # RGB channel image format data
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert RGB channel image format data to Tensor
        # channel image format data
        lr_tensor = imgproc.image2tensor(lr_image, False, False).unsqueeze_(0)

        # Transfer Tensor channel image format data to target device
        lr_tensor = lr_tensor.to(device=DEVICE)

        with torch.no_grad():
            sr_tensor = self.model(lr_tensor)

        transform = T.ToPILImage()

        # Remove batch dimension
        sr_tensor.squeeze_(0)
        return transform(sr_tensor)

    def build_model(self):
        WEIGHTS_PATH = "./weights/SRGAN_x4-ImageNet-c71a4860.pth.tar"
        DEVICE = torch.device("cpu")

        # Initialize the model
        model = Generator()
        model = model.to(memory_format=torch.channels_last, device=DEVICE)
        print("Build SRGAN model successfully.")

        # Load the SRGAN model weights
        checkpoint = torch.load(WEIGHTS_PATH, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Load SRGAN model weights `{WEIGHTS_PATH}` successfully.")

        # Start the verification mode of the model.
        model.eval()

        return model


def your_streamlit_app(lightning_app_state):
    static_text = """
    # SRGAN Lightning App

    This is a simple [Lightning app](https://lightning.ai) that runs
    SRGAN model based on [this](https://github.com/Lornatang/SRGAN-PyTorch)
    GitHub repository.

    If you want to learn more about Lightning Apps, checkout the official
    [lightning.ai](https://lightning.ai) website.

    If you have any questions or suggestions, you can find
    me [here](http://sebastianraschka.com) and
    [here](http://twitter.com/rasbt).
    """
    st.write(static_text)


class ChildFlow(L.LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=your_streamlit_app)


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.demo = SRGAN(cloud_compute=L.CloudCompute("cpu", 1))
        self.about_page = ChildFlow()

    def run(self):
        self.demo.run()

    def configure_layout(self):
        tab_1 = {"name": "SRGAN Demo", "content": self.demo}
        tab_2 = {
            "name": "SRGAN Paper",
            "content": "https://arxiv.org/pdf/1609.04802v5.pdf",
        }
        tab_3 = {"name": "About", "content": self.about_page}
        return tab_1, tab_2, tab_3


app = L.LightningApp(RootFlow())
