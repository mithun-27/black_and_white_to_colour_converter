import requests
import os

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
        
    models = {
        "colorization_release_v2.caffemodel": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
        "colorization_deploy_v2.prototxt": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt",
        "pts_in_hull.npy": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"
    }
    
    for filename, url in models.items():
        dest = os.path.join("models", filename)
        # Always retry weights if it's too small
        if not os.path.exists(dest) or os.path.getsize(dest) < 100000: 
            download_file(url, dest)
        else:
            print(f"{filename} already exists.")
