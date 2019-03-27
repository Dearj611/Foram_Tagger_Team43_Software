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
    model_path = Model.get_model_path('foram-resnet18', _workspace=ws)
    model = load_checkpoint(model_path)


def load_checkpoint(path):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 17)
    model.load_state_dict(torch.load(path))
    model.idx_to_class = {num:species for num,species in enumerate(['G-crassaformis', 'G-elongatus', 'G-hexagonus', 'G-ruber', 'G-ruber pink', 'G-sacculifer', 'G-scitula', 'G-siphonifera', 'G-truncatulinoides', 'G-tumida', 'G-ungulata', 'N-acostaensis', 'N-dutertrei', 'N-humerosa', 'O-universa', 'P-obliquiloculata', 'S-dehiscen'])}
    return model