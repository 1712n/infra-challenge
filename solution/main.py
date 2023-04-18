from loguru import logger
import requests
from os import environ

CONTESTANT_NAME = environ.get("CONTESTANT_NAME", "Unknown")

if __name__ == "__main__":
    logger.info(f"Starting the application for {CONTESTANT_NAME}...")

    response = requests.get("https://httpbin.org/status/418")
    logger.info(f"Got response: {response}")

    logger.info("Exiting the application...")
