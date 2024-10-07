import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image



RAW_SAVE_DIR = "/Users/gayusri/GeoCracker/data/raw/images"
PROC_SAVE_DIR = "/Users/gayusri/GeoCracker/data/processed/images"

if not os.path.exists(RAW_SAVE_DIR):
    os.makedirs(RAW_SAVE_DIR)

def scrape_bollard_images(url):
    service = Service(executable_path="/Users/gayusri/Downloads/chromedriver-mac-x64 2/chromedriver")  
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--no-sandbox") 
    chrome_options.add_argument("--disable-dev-shm-usage") 
    chrome_options.add_argument("--window-size=1920x1080")  
    
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)

    for _ in range(2):
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    images = soup.find_all("img")
    
    for idx, img in enumerate(images):
        img_url = img.get("src")
        if img_url is None: 
            print(f"No 'src' attribute found for image {idx + 1}. Skipping...")
            continue  

        if img_url.startswith("data:image/jpeg;base64,"):
            print(f"Image {idx + 1} is a base64 string, skipping download...")
            continue 


        if not img_url.startswith("http"):
            img_url = urljoin(url, img_url)
            if not img_url.startswith("http"):
                print(f"Image {idx + 1} has an invalid URL format: {img_url}, skipping...")
                continue 

        img_filename = f'bollard_{idx}.jpg'
        img_path = os.path.join(RAW_SAVE_DIR, img_filename)

        if os.path.exists(img_path):
            print(f"Image {img_filename} already exists, skipping download...")
            continue 

        try:
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Downloaded image {idx + 1} to {img_path}")
        except Exception as e:
            print(f"Error downloading image {idx + 1}: {e}")


    driver.quit()

def preprocess_images(save_dir):
    processed_dir = PROC_SAVE_DIR
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for img_file in os.listdir(save_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(save_dir, img_file)
            try:
                img = Image.open(img_path)
                img = img.resize((224, 224))  # Resize to 224x224
                img.save(os.path.join(processed_dir, img_file))
                print(f"Processed {img_file}")
            except Exception as e:
                print(f"Could not process {img_file}: {e}")

    
if __name__ == "__main__":
    scrape_bollard_images("https://www.google.com/search?q=austrian+bollards+pictures&tbm=isch")
    preprocess_images(RAW_SAVE_DIR)
