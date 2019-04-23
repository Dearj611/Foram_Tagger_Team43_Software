Foram Tagger

To setup virtual environment:
python3 -m virtualenv venv
source venv/bin/activate

For homebrew version of mysql, run these two lines to set up database:

sudo chown -R _mysql:mysql /usr/local/var/mysql
sudo mysql.server start

To ensure we are using the same version of libraries:
pip install -r requirements.txt
