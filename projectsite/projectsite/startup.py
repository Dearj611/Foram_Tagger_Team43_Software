# The model is made global so that it can later be used in the django app
import os
from azureml.core.model import Model
from azureml.core import Workspace
import torch
from torch import nn
from torchvision import models
from azureml.core.authentication import ServicePrincipalAuthentication


def run():
    global model
    svc_pr = ServicePrincipalAuthentication(
        tenant_id="1faf88fe-a998-4c5b-93c9-210a11d9a5c2",
        service_principal_id="3683e499-d9d1-4b25-9e51-fc0c056415da",
        service_principal_password=os.environ.get("AZUREML_PASSWORD"))
    # retrieve the path to the model file using the model name
    ws = Workspace.get(name='foram-workspace',
                       subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b',
                       auth=svc_pr)
    model_path = Model.get_model_path('resnet_cv', _workspace=ws)
    model = load_checkpoint(model_path)


def load_checkpoint(path):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 16)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.idx_to_class = {0: 'G-crassaformis', 1: 'G-elongatus', 2: 'G-hexagonus', 3: 'G-ruber', 4: 'G-sacculifer', 5: 'G-scitula', 6: 'G-siphonifera', 7: 'G-truncatulinoides', 8: 'G-tumida', 9: 'G-ungulata', 10: 'N-acostaensis', 11: 'N-dutertrei', 12: 'N-humerosa', 13: 'O-universa', 14: 'P-obliquiloculata', 15: 'S-dehiscen'}
    return model