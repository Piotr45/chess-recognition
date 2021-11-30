import shutil

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

import time
import threading
import os

CHROMEDRIVER_EXE_PATH = "C:/Program Files (x86)/chromedriver.exe"
EXTRA_DATA_FOLDER_PATH = "extra_data"
FIGURES = ["pawn", "rook", "knight", "bishop", "king", "queen"]


def gen_first_n_figure_pictures_from_google_images(n: int, figure: str) -> None:
    def _do_scrolling(_driver: webdriver.Chrome) -> None:
        last_height = _driver.execute_script('return document.body.scrollHeight')
        while True:
            _driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            time.sleep(2)
            new_height = _driver.execute_script('return document.body.scrollHeight')
            try:
                _driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
                time.sleep(2)
            except Exception as exc:
                print(exc)
            if new_height == last_height:
                break
            last_height = new_height

    def _make_screenshot_and_put_under_label(_driver: webdriver.Chrome, image_number: int, _figure: str) -> None:
        _driver.find_element(By.XPATH, f'//*[@id="islrg"]/div[1]/div[{image_number}]/a[1]/div[1]/img').click()
        _driver.find_element(
            By.XPATH,
            '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img'). \
            screenshot(f'./extra_data/{_figure.capitalize()}/{_figure}{image_number:03d}.png')

    service = Service(CHROMEDRIVER_EXE_PATH)
    driver = webdriver.Chrome(service=service)
    driver.get('https://www.google.pl/imghp?hl=pl&ogbl')

    time.sleep(3)

    agree_clause = driver.find_element(By.ID, "L2AGLb")
    agree_clause.send_keys(Keys.ENTER)

    time.sleep(3)

    box = driver.find_element(By.NAME, 'q')
    box.send_keys(f"{figure} chess jpg")
    box.send_keys(Keys.ENTER)

    time.sleep(3)

    _do_scrolling(driver)

    for i in range(1, n):
        try:
            _make_screenshot_and_put_under_label(driver, i, figure)
        except Exception as ex:
            print(ex)


def handle_directories() -> None:
    if os.path.exists(EXTRA_DATA_FOLDER_PATH):
        shutil.rmtree(EXTRA_DATA_FOLDER_PATH)

    os.makedirs(EXTRA_DATA_FOLDER_PATH)

    for fig in FIGURES:
        os.makedirs(f"{EXTRA_DATA_FOLDER_PATH}/{fig.capitalize()}")


if __name__ == '__main__':
    handle_directories()

    threads = []

    for index, _figure in enumerate(FIGURES):
        thread = threading.Thread(target=gen_first_n_figure_pictures_from_google_images, args=(250, _figure,))
        threads.append(thread)
        print(f"Started thread {index} for {_figure}")
        thread.start()

    for index, thread in enumerate(threads):
        thread.join()
        print(f"Thread {index} finished")
