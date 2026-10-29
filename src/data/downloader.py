"""
Dataset Downloader and Extractor for MetroPT-3
Author: Batch 11 - Predictive Maintenance System
Dataset Source: UCI Machine Learning Repository (ID: 791) / Davari et al.
"""

import os
import sys
import zipfile
import urllib.request
import time

UCI_METROPT3_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
ZIP_FILE_PATH = os.path.join(RAW_DATA_DIR, "metropt_3_dataset.zip")
TARGET_CSV_NAME = "MetroPT3(AirCompressor).csv"
TARGET_CSV_PATH = os.path.join(RAW_DATA_DIR, TARGET_CSV_NAME)


def download_progress_hook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading: {progress_size / (1024 * 1024):.1f} MB [{percent}%] at {speed} KB/s")
    sys.stdout.flush()


def download_and_extract(force_download=False):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Check if CSV already exists
    if os.path.exists(TARGET_CSV_PATH) and not force_download:
        file_size_mb = os.path.getsize(TARGET_CSV_PATH) / (1024 * 1024)
        print(f"[OK] MetroPT-3 CSV already exists at: {TARGET_CSV_PATH} ({file_size_mb:.2f} MB)")
        return TARGET_CSV_PATH

    # Download zip if not present
    if not os.path.exists(ZIP_FILE_PATH) or force_download:
        print(f"Downloading MetroPT-3 dataset from UCI Repository: {UCI_METROPT3_URL}")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)

        try:
            urllib.request.urlretrieve(UCI_METROPT3_URL, ZIP_FILE_PATH, reporthook=download_progress_hook)
            print("\n[OK] Download complete!")
        except Exception as e:
            print(f"\n[ERROR] Failed to download dataset: {e}")
            raise e

    # Extract zip
    print(f"Extracting zip archive to: {RAW_DATA_DIR}...")
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
        print("[OK] Extraction complete! Files in directory:")
        for fname in zip_ref.namelist():
            print(f"  - {fname}")

    # Verify CSV file existence
    # Sometimes zip has nested or slightly different file names
    csv_candidates = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.csv')]
    if csv_candidates:
        actual_csv = os.path.join(RAW_DATA_DIR, csv_candidates[0])
        print(f"[SUCCESS] Ready! Dataset CSV located at: {actual_csv} ({os.path.getsize(actual_csv)/(1024*1024):.2f} MB)")
        return actual_csv
    else:
        raise FileNotFoundError(f"Could not find extracted CSV in {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_and_extract()
