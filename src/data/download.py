import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CMAPSS_URL = "https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip"
DATA_DIR = "data"

def download_cmapss_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    zip_path = os.path.join(DATA_DIR, "CMAPSSData.zip")
    
    # Check if we already have the unzipped files
    required_files = [
        "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
        "test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt",
        "RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt"
    ]
    
    has_all_files = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in required_files)
    
    if has_all_files:
        logging.info("CMAPSS dataset already exists and is complete in 'data/' directory.")
        return

    if not os.path.exists(zip_path):
        logging.info(f"Downloading CMAPSS dataset from {CMAPSS_URL}...")
        try:
            # We use a user agent because some servers block python urllib
            req = urllib.request.Request(CMAPSS_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            logging.info("Download completed.")
        except Exception as e:
            logging.error(f"Failed to download from NASA repository: {e}")
            logging.info("Please download CMAPSSData.zip manually and place it in the 'data/' folder.")
            return

    if not zipfile.is_zipfile(zip_path):
        logging.error("The downloaded file is not a valid ZIP archive.")
        logging.error("NASA's website may be redirecting the download or requiring a login.")
        logging.info("==== MANUAL DOWNLOAD REQUIRED ====")
        logging.info("1. Go to https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        logging.info("2. Download the 'Turbofan Engine Degradation Simulation Data Set' (CMAPSSData.zip)")
        logging.info("3. Place 'CMAPSSData.zip' directly inside the 'data' folder.")
        logging.info("==================================")
        
        # Remove the invalid file so it doesn't block future attempts
        os.remove(zip_path)
        return

    logging.info("Extracting CMAPSSData.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    logging.info("Extraction complete. All datasets FD001-FD004 are ready.")

if __name__ == "__main__":
    download_cmapss_data()
