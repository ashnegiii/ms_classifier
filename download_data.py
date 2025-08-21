from pathlib import Path

import gdown
import requests

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

folder_url = "https://drive.google.com/drive/folders/1qLHdl54UUeC67oD9Wjawz3m8OD8xs31x?usp=sharing"

gdown.download_folder(folder_url, output=str(output_dir), quiet=False, use_cookies=False)