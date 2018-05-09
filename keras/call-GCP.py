import googleapiclient.discovery
from oauth2client.client import GoogleCredentials



PROJECT_ID = "deep-learning-198500"
MODEL_NAME = "earnings"
CREDENTIALS_FILE = "./keras/ml-credentials.json"

# These are the values we want a prediction for
inputs_for_prediction = [
    {"inputs": [0.4999, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5]}
]

# Connect to the Google Cloud-ML Service
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
#credentials = service_account.client.credentials_from_clientsecrets_and_code()
service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)

# Connect to Prediction Model
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
response = service.projects().predict(
    name=name,
    body={'instances': inputs_for_prediction}
).execute()

# Report any errors
if 'error' in response:
    raise RuntimeError(response['error'])

# Grab the results from the response object
results = response['predictions']

# Print the results!
print(results)
