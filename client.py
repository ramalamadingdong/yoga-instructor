import requests
import time

SERVER_URL = "http://localhost:5000/get_position"


def get_and_print_position():
    response = requests.get(SERVER_URL)
    position = response.json()
    return position

while True:
    print(f"Waiting for human to assume {get_and_print_position()}...")
    time.sleep(5)  # Check every second
    get_and_print_position()

