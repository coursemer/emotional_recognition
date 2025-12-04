import requests
import os

# --- Configuration ---
MODEL_URL = "https://github.com/oarriaga/face_classification/raw/master/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
MODEL_FILE = "fer2013_mini_XCEPTION.102-0.66.hdf5"
OPENCV_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
OPENCV_FILE = "haarcascade_frontalface_default.xml"

def download_file(url, filename):
    """Downloads a file from a URL and saves it locally."""
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Skipping download.")
        return

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"'{filename}' downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        print("Please check your internet connection or download the file manually.")

if __name__ == "__main__":
    download_file(MODEL_URL, MODEL_FILE)
    download_file(OPENCV_URL, OPENCV_FILE)
    print("\nSetup complete. You can now run 'python main.py'")
