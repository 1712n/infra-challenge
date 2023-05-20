import random
from pathlib import Path

from locust import FastHttpUser, task

ROOT_DIR = Path(__file__).parent.absolute()
path_to_examples = ROOT_DIR / Path("examples/examples.txt")


TEXTS = []
with open(path_to_examples, "r") as f:
    for line in f.readlines():
        TEXTS.append(line.rstrip("\n"))


class TechUser(FastHttpUser):
    @task
    def process(self):
        text = TEXTS[random.randint(0, len(TEXTS) - 1)]
        self.client.post(
            url="/process",
            json=text,
            headers={"Content-Type": "application/json"},
        )
