import os
import time
import yaml
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress Selenium logs
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('webdriver_manager').setLevel(logging.WARNING)

os.environ['WDM_LOG_LEVEL'] = '0'


def get_nmr_prediction(url, label, cas_id, smiles, output_folder):
    driver = None
    try:
        logging.info(f"Starting NMR prediction for {cas_id} ({label})")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)

        # Wait for the privacy information dialog to be visible
        WebDriverWait(driver, 6).until(
            EC.visibility_of_element_located((By.XPATH, "//button[text()='I agree']"))
        )

        
        # Click the "I agree" button
        agree_button = driver.find_element(By.XPATH, "//button[text()='I agree']")
        agree_button.click()

        # Wait for potential dynamic content loading
        time.sleep(5)

        # Locate the NMR data by the 'contenteditable' attribute within the 'ci-module-content' class
        nmr_results_container = driver.find_element(By.CSS_SELECTOR, '.ci-module-content [contenteditable="true"]')
        nmr_results = nmr_results_container.text

        if not nmr_results.strip():
            logging.warning(f"No NMR data found for {cas_id} ({label}). Skipping...")
            return None, None

        # Click the "Show fullscreen" button on the bottom right
        WebDriverWait(driver, 7).until(
            EC.element_to_be_clickable((By.XPATH, "//li[@title='Show fullscreen']"))
        )
        full_screen_button = driver.find_element(By.XPATH, "//li[@title='Show fullscreen']")
        full_screen_button.click()

        # Wait for the page to fully load after entering full screen
        time.sleep(4)

        # Take a screenshot and save it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        screenshot_filename = os.path.join(output_folder, f"{cas_id}_{label.replace(' ', '_').lower()}.png")
        driver.save_screenshot(screenshot_filename)
        logging.info(f"Screenshot saved to: {screenshot_filename}")

        return nmr_results, screenshot_filename
    except Exception as e:
        logging.error(f"An error occurred while processing {cas_id} for {label}: {e}")
        return None, None
    finally:
        if driver:
            driver.quit()
            logging.info(f"Closed the browser for {cas_id} ({label})")
            time.sleep(0.5)
    

def load_yaml(filename):
    if not os.path.exists(filename):
        logging.warning(f"YAML file {filename} does not exist.")
        return {}
    with open(filename, 'r') as file:
        logging.info(f"Loading YAML file {filename}")
        return yaml.safe_load(file) or {}

def save_yaml(data, filename):
    with open(filename, 'w') as file:
        yaml.dump(data, file)
        logging.info(f"Saved data to YAML file {filename}")

def main(yaml_file, results_file, progress_file, base_output_folder, safety_file):

    data = load_yaml(yaml_file)
    last_scraped_cas = load_yaml(progress_file) if os.path.exists(progress_file) else None
    results = load_yaml(results_file) if os.path.exists(results_file) else {}

    start_scraping = not last_scraped_cas

    for cas_id, molecule_data in data.items():
        if not start_scraping:
            if cas_id == last_scraped_cas:
                start_scraping = True
            continue

        smiles = molecule_data.get('SMILES')
        if not smiles:
            logging.warning(f"No SMILES data for {cas_id}. Skipping...")
            continue

        urls = {
            "1H NMR": f"https://www.nmrdb.org/service.php?name=nmr-1h-prediction&smiles={smiles}",
            "13C NMR": f"https://www.nmrdb.org/service.php?name=nmr-13c-prediction&smiles={smiles}"
        }

        molecule_results = {}
        screenshot_paths = {}
        all_successful = True
        for name, url in urls.items():
            output_folder = os.path.join(base_output_folder, name.replace(' ', '_'))
            nmr_results, screenshot_path = get_nmr_prediction(url, name, cas_id, smiles, output_folder)
            if nmr_results:
                molecule_results[name] = nmr_results
                screenshot_paths[name] = screenshot_path
            else:
                all_successful = False
                break

        if all_successful:
            results[cas_id] = {
                "NMR Results": molecule_results,
                "Screenshots": screenshot_paths
            }

            save_yaml(results, results_file)
            logging.info(f"DO NOT CLOSE YET.... saving to YAML file")
            save_yaml(cas_id, progress_file)
            logging.info(f"DO NOT CLOSE YET...... saving to progress")

            save_yaml(results, safety_file)
            logging.info(f"Successfully scraped CAS ID: {cas_id}. You may close now.")
        else:
            logging.warning(f"Skipping CAS ID: {cas_id} due to errors or empty NMR data")
    logging.info("Finished")


# Example usage
yaml_file = '/Users/rudrasondhi/Desktop/Specto 0.2/Specto-0.2/Data/All SMILES, SELFIES, Taut.yaml'  # Replace with the path to your YAML file
results_file = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/NMR Data/nmr_results.yaml'
progress_file = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/NMR Data/progress.yaml'
safety_file = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/NMR Data/Safety.yaml'
base_output_folder = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Screenshots'  # Specify the base folder for saving screenshots
main(yaml_file, results_file, progress_file, base_output_folder, safety_file)


"""


from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

def get_nmr_prediction(url, label, smiles):
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)

        # Wait for the privacy information dialog to be visible
        WebDriverWait(driver, 6).until(
            EC.visibility_of_element_located((By.XPATH, "//button[text()='I agree']"))
        )

        # Click the "I agree" button
        agree_button = driver.find_element(By.XPATH, "//button[text()='I agree']")
        agree_button.click()

        # Wait for potential dynamic content loading
        time.sleep(6)

        # Locate the NMR data by the 'contenteditable' attribute within the 'ci-module-content' class
        nmr_results_container = driver.find_element(By.CSS_SELECTOR, '.ci-module-content [contenteditable="true"]')
        nmr_results = nmr_results_container.text



        # Click the "Show fullscreen" button on the bottom right
        WebDriverWait(driver, 6).until(
            EC.element_to_be_clickable((By.XPATH, "//li[@title='Show fullscreen']"))
        )
        full_screen_button = driver.find_element(By.XPATH, "//li[@title='Show fullscreen']")
        full_screen_button.click() 

        # Wait for the page to fully load after entering full screen
        time.sleep(3)

        # Click the "Download as SVG vector file" button
        WebDriverWait(driver, 6).until(
            EC.element_to_be_clickable((By.XPATH, "//li[@title='Download as SVG vector file']"))
        )
        download_svg_button = driver.find_element(By.XPATH, "//li[@title='Download as SVG vector file']")
        download_svg_button.click()

        # Wait for the download to complete
        time.sleep(8)

        # Move the downloaded file to the desired location with a unique name
        download_path = os.path.expanduser('~/Downloads')
        svg_filename = f"{smiles}_{label.replace(' ', '_').lower()}.svg"
        initial_svg_path = os.path.join(download_path, 'graph.svg')
        final_svg_path = os.path.join(download_path, svg_filename)
        if os.path.exists(initial_svg_path):
            os.rename(initial_svg_path, final_svg_path)

        return nmr_results
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        if driver:
            driver.quit()

def main(smiles):
    # Define the NMR prediction URLs
    urls = {
        "1H NMR": f"https://www.nmrdb.org/service.php?name=nmr-1h-prediction&smiles={smiles}",
        "13C NMR": f"https://www.nmrdb.org/service.php?name=nmr-13c-prediction&smiles={smiles}"
    }

    for name, url in urls.items():
        nmr_results = get_nmr_prediction(url, name, smiles)
        
        # Print the results
        print(f"--- {name} Prediction Results ---")
        print(nmr_results)
        print("\n")

# Example SMILES string to run the function
smiles_string = 'c1ccccc1CC'  # Replace with any valid SMILES
main(smiles_string)

"""
