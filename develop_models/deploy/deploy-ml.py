from azureml.core import Workspace
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice
from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.image import Image

ws = Workspace.create(name='foram-workspace',
                      subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b',	
                      resource_group='foram-tagger-ml',
                      create_resource_group=True,
                      location='northeurope' 
                      )
ws = Workspace.get(name='foram-workspace', subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b')
ws.get_details()
ws.write_config()
model = Model.register(model_path="./model1/resnet18.pth",
                       model_name="resnet18",
                       description="inference",
                       workspace=ws)

# Image configuration
image_config = ContainerImage.image_configuration(execution_script="score.py",
                                                  runtime="python",
                                                  conda_file="myenv.yml"
                                                  )
image = Image()                                       
# Setup Container Instance                                
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=4,
                                               tags={"data": "foram", "type": "classification"}, 
                                               description='Species Categorization')
# Actual Deployment                                               
service = Webservice.deploy_from_model(deployment_config=aciconfig,
                                       image_config=image_config,
                                       name='foram-tagger-infer',
                                       models=[model],
                                       workspace=ws)
service.wait_for_deployment(show_output=True)
service = Webservice(ws, 'foram-tagger-inference')
service.update(image = image_config)
model = Model.get_model_path()
service.wait_for_deployment(show_output = True)

# Register the image from the image configuration. This allows you to deploy once, use everywhere
# This method of deployment also allows you to easily update the webservice's image
image = ContainerImage.create(name="foramimage",
                              models=[model], #this is the model object
                              image_config=image_config,
                              workspace=ws
                              )
image.wa
service.image
service = Webservice.deploy_from_image(deployment_config=aciconfig,
                                       image=image,
                                       name='foram-tagger-inference',
                                       workspace=ws)
service.wait_for_deployment(show_output=True)

def deploy_model(path, model_name):
    ws = Workspace.get(name='foram-workspace', subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b')
    model = Model.register(model_path=path,
                       model_name=model_name,
                       description="inference",
                       workspace=ws)