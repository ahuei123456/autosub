import os
from dotenv import load_dotenv

load_dotenv()

GCS_BUCKET = os.environ.get("AUTOSUB_GCS_BUCKET")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

# Google Cloud APIs generally read GOOGLE_APPLICATION_CREDENTIALS automatically
