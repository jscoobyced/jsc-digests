from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv
import os

load_dotenv()

login(os.environ["HUGGINGFACE_TOKEN"])

# Download SparkAudio model
snapshot_download(
    os.environ["SPARK_AUDIO_MODEL"],
    local_dir="models/" + os.environ["SPARK_AUDIO_MODEL"],
)
