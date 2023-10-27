from google.cloud import aiplatform
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="scenic-aileron-389223-00fedf4349a3.json"

PROJECT_ID = "scenic-aileron-389223"


aip_endpoint_name = (
    f"projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/imagegeneration:predict"
)

print(aip_endpoint_name)

endpoint = aiplatform.Endpoint(aip_endpoint_name)

data = {
  "instances": [
    {
      "prompt": "Photo centered of a full length man dressing blue t-shirt with light jeans and white snickers"
    }
  ],
  "parameters": {
    "sampleCount": 1
  }
}

instances = [data]

results = endpoint.predict(instances=instances)

print(results.predictions)