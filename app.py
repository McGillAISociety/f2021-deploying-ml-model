import io
import torch
from flask import Flask, render_template, request
from model import Model, transformation
from PIL import Image

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


model = Model()
model.load_state_dict(torch.load("lenet_epoch=12_test_acc=0.991.pth"))


@app.route("/data", methods=["POST"])
def data():
    im = Image.open(io.BytesIO(request.data)).convert("L")
    im = transformation(im).unsqueeze(0)
    with torch.no_grad():
        preds = model(im)
        preds = torch.argmax(preds, axis=1)
        print(preds[0].item())
        return {"data": preds[0].item()}
    return {"data": 1 + 1}
