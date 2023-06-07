import git
import logging
import sys

from config.config import settings


log = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(10)
formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(10)


def download_HF_models(models_settings):
    # Clone a remote repository
    for model in models_settings["models"]:
        model_name = model.values()[0].to_dict()['model_name']
        repo_url = f"https://huggingface.co/{model_name}"
        local_path = f"/models/{model_name}"
        repo = git.Repo.clone_from(repo_url, local_path)
        log.debug(f'Repository Cloned at location: {local_path}')


if __name__ == "__main__":
    download_HF_models(settings)