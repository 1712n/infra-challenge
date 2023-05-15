import os
from pathlib import Path
import yaml

from dotenv import load_dotenv

load_dotenv()

DEPLOY_MODE = os.getenv("DEPLOY_MODE")

if DEPLOY_MODE == "DEV":
    ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
    WORKSPACE_PATH = ROOT_DIR / Path("solution/workspace")
elif DEPLOY_MODE == "PROD":
    ROOT_DIR = Path("/app")
    WORKSPACE_PATH = Path("/workspace")
else:
    # TODO log error setup
    pass

MODELS_LIST_CONF_PATH = WORKSPACE_PATH / Path("models_list.yaml")

with open(MODELS_LIST_CONF_PATH, 'r') as stream:
    models_dict = yaml.safe_load(stream)
