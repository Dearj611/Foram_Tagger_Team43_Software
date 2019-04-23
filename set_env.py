# importing the requests library
# This file only works when running in the VM
import requests
import os

# Step 1: Fetch an access token from an MSI-enabled Azure resource      
# Note that the resource here is https://vault.azure.net for the public cloud, and api-version is 2018-02-01
MSI_ENDPOINT = "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net"
r = requests.get(MSI_ENDPOINT, headers = {"Metadata" : "true"})

# Extracting data in JSON format 
# This request gets an access token from Azure Active Directory by using the local MSI endpoint
data = r.json()
# print(data)

# Step 2: Pass the access token received from the previous HTTP GET call to the key vault
KeyVaultURL = "https://vm-secrets.vault.azure.net/secrets/?api-version=2016-10-01"
all_secrets = requests.get(url = KeyVaultURL, headers = {"Authorization": "Bearer " + data["access_token"]})

for value in all_secrets.json()["value"]:
    url = value["id"] + "/?api-version=2016-10-01"
    env_var = os.path.basename(value["id"]).replace('-', '_')
    secret = requests.get(url=url, headers = {"Authorization": "Bearer " + data["access_token"]})
    os.environ[env_var] = secret.json()["value"]
    print(env_var+ ' set to ' + secret.json()["value"])