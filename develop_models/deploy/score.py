from azureml.core.model import Model
from azureml.core import Workspace
import numpy as np
import json
import torch
from torch import nn
from torchvision import models, transforms
# from PIL import Image


def init():
    global model
    # retrieve the path to the model file using the model name
    ws = Workspace.get(name='foram-workspace', subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b')
    model_path = Model.get_model_path('foram-resnet18', _workspace=ws)
    model = load_checkpoint(model_path)


def run(raw_data):
    '''
    Must accept a json, must return a json
    '''
    try:
        to_return = {}
        data = list(json.loads(raw_data)['data'])
        for i, img in enumerate(data):
            img = np.array(img, dtype='uint8')
            top_p, top_classes, true_class = predict(model, img)
            to_return[i] = {'top_p':top_p.tolist(), 'top_classes': top_classes, 'true_class': true_class}
        to_return = json.dumps({'data': to_return})
        return to_return
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})


def process_image(image):
    """Process an image into a PyTorch tensor"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(image)
    return img


def predict(model, image, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns
    top_p: the probabilites
    top_classes: top 5 classes
    """

    # Convert to pytorch tensor
    img_tensor = process_image(image)

    # Resize
    if torch.cuda.is_available():
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return top_p, top_classes, top_classes[0]


def load_checkpoint(path):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 17)
    model.load_state_dict(torch.load(path))
    model.idx_to_class = {num:species for num,species in enumerate(['G. crassaformis', 'G. elongatus', 'G. hexagonus', 'G. ruber', 'G. ruber pink', 'G. sacculifer', 'G. scitula', 'G. siphonifera', 'G. truncatulinoides', 'G. tumida', 'G. ungulata', 'N. acostaensis', 'N. dutertrei', 'N. humerosa', 'O. universa', 'P. obliquiloculata', 'S. dehiscen'])}
    return model
