from azureml.core.conda_dependencies import CondaDependencies

myenv = CondaDependencies()
myenv.add_pip_package('torch==1.0.0')
myenv.add_pip_package('torchvision==0.2.1')
myenv.add_pip_package('numpy==1.15.4')

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())