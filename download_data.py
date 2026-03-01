from pathlib import Path
import shutil

import gdown

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

folder_url = "https://drive.google.com/drive/folders/1qLHdl54UUeC67oD9Wjawz3m8OD8xs31x?usp=sharing"

gdown.download_folder(folder_url, output=str(output_dir), quiet=False, use_cookies=False)


# Move models to visualizer/models (single location). Remove data/raw/models afterwards.
visualizer_models = Path("visualizer/models")
downloaded_models = output_dir / "models"
if downloaded_models.is_dir():
    if not visualizer_models.exists() or not any(visualizer_models.iterdir()):
        visualizer_models.mkdir(parents=True, exist_ok=True)
        for p in downloaded_models.iterdir():
            dest = visualizer_models / p.name
            if p.is_file():
                shutil.move(str(p), str(dest))
            elif p.is_dir():
                shutil.move(str(p), str(dest))
    shutil.rmtree(downloaded_models)