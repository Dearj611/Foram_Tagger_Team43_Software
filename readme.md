Foram Tagger

To setup virtual environment:
python3 -m virtualenv venv
source venv/bin/activate

For homebrew version of mysql, run these two lines to set up database:

sudo chown -R _mysql:mysql /usr/local/var/mysql
sudo mysql.server start

To ensure we are using the same version of libraries:
pip install -r requirements.txt

To run our image_segmentation.py, opencv and re, solution:
pip install opencv-python
change "from regex as re" to "import re"
pip install matplotlib


enviroment set up: 
export DJANGO_SETTINGS_MODULE=projectsite.production_settings
export SECRET_KEY='az7u5-i_ot%*#l9dhyx%cnk0$b&8h0icne@pm4qd1)s*9s-%1z'
export DB_HOST='foramdatabase.mysql.database.azure.com'
export DB_PASSWORD='ForamTagger123'
export DB_USER='camelcars@foramdatabase'
export AZ_STORAGE_ACCOUNT_NAME='forampics'
export AZ_STORAGE_CONTAINER='allstaticfiles'
export AZ_STORAGE_KEY='4nwt5cexYaNCgmsk5NrLLm5lmRprYobFVepz+hhb6b7hv2f6zifM1EPmoqT7SMTsUYvWSe3nREd/dS6g8Thjmg=='
export AZ_IMAGE_CONTAINER='media'
export AZUREML_PASSWORD='R5q6DTJsLHmxPTYcegeXMQ/2I9pABbkRu9ru7h1Srtc='
