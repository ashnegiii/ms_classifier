# Muppet Show Classifier

This repository consists of four deliverables:

1. **Training pipeline** (**/src** folder)— Train multi-label classifiers (EfficientNet-B2, ResNet-50, ConvNeXt-Tiny, CLIP ViT-B/16) on Muppet Show data with configurable episode splits, augmentation, and evaluation.
2. **Video prediction** (**/video_prediction** folder) — Run a trained model on a video to produce a **CSV file** (per-frame, per-character predictions) in the same folder as the video, for use in future lectures!
3. **Visualizer** (**/visualizer** folder) — Additionally a web app was provided to showcase **Grad-CAM** and **predictions on each frame** using different models: load a video, pick a frame, select characters, and see confidence scores plus heatmaps.
4. **Provided ground truth predictions** (**/result_ground_truth/** folder) — The ground truth predictions from the best model for each episode are stored in that folder. For every episode, the best model was always the one that did not train on that specific episode but used it as a test episode for producing the corresponding CSV ground truth file.


## Contents

| Component                                         | Description                                                                                                                                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Source code for the training pipeline**         | Full project code: `src/` (training, evaluation, backbones, Grad-CAM), `visualizer/` (backend API + frontend), `download_data.py`, `generate_data.py`, configs under `src/config/`.                    |
| **Training data**                                 | Raw data in `data/raw/` (after running `download_data.py` (see below)): videos/episodes (02-01-01, 03-04-03, 02-04-04, miss-piggy, the-cook, rowlf-the-dog, etc.) and associated annotation/CSV files. |
| **Models**                                        | Trained checkpoints (`.pth`) in `visualizer/models/`, fetched and placed there by `download_data.py`. |
| **Model-based predictions (result_ground_truth)** | For each episode, ground truth predictions from the best model that excluded that episode from training and used it as test, stored as CSVs in `result_ground_truth/`.                                 |


Supported model architectures: **EfficientNet-B2**, **ResNet-50**, **ConvNeXt-Tiny** and **CLIP ViT-B/16**. The best models with the best configuration from the experiments were selected. The models were trained on all available videos **except** **03-04-17**, which was held out for testing.

---

# Setup and Running the Visualizer

A visualizer web app was created to showcase the trained deep learning models for future lectures. Since the model checkpoints and all videos require a lot of memory, a download script was created to download all necessary files for running the training pipeline, as well as to download the best models for inference.
**First, download all data and models (once).** 

Run from the repository root:

```bash
python download_data.py
```

This fetches the dataset into `data/raw` (which were used for the training/test pipeline) and also downloads the final trained **models** folder from Google Drive into `visualizer/models` (only if that folder doesn’t already exist). Do this before training or running the visualizer. The visualizer is used for showing the Predictions and Grad-CAM implementation for future lectures.

## Prerequisites

- **Python 3.12** with `pip` was used throughout the project.
- **Node.js** and **npm** (for the frontend)

### Backend (FastAPI)

From the **repository root**, create and activate a virtual environment, then install dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:
source venv/Scripts/activate
# On macOS/Linux:
# source venv/bin/activate

# Install Python dependencies (once)
pip install -r requirements.txt

# Run the API
PYTHONPATH=. uvicorn visualizer.server:app --reload --port 8000
```

- This will start the backend on port 8000: **[http://localhost:8000](http://localhost:8000)**

**Model folder:** After running `download_data.py`, the four models (`.pth` files) are in **`visualizer/models`**. In the frontend app, choose **Architecture** and click **Load model** — the server loads the matching file from that folder automatically.

### 2. Frontend (Visualizer)

In a **new terminal**, from the repository root:

```bash
cd visualizer/frontend
npm install
npm run dev
```

- The app runs at **[http://localhost:5173](http://localhost:5173)**

Start the backend first so that “Load model" and “Create prediction" work correctly.

Then start the frontend (`npm run dev`).

---

## How the Visualizer Works

The **Muppet Show Prediction Viewer** lets you load a trained model and a video, pick a frame, choose which characters to predict, and then see per-character prediction scores and also the Grad-CAM heatmaps.

How to use the Web App when started:

1. **Load the model**
  In the **Model** section, choose **Architecture** and click **Load model**. The server loads the matching file from `models/` and prepares Grad-CAM. When done, a success toast appears.
2. **Load video**
  Click “Load Video" and pick a video file. The video plays in the browser. You can adjust with the slider or step frame-by-frame. Currently only .mp4 files are supported!
3. **Select characters**
  Use the checkboxes to select which of the six characters (Kermit, Miss Piggy, Swedish Chef, Fozzie Bear, Statler & Waldorf, Rowlf The Dog) you want predictions and Grad-CAMs for.
4. **Create prediction**
  Move to the desired frame, then click **“Prediction erstellen"**. The frontend:
  - Sends `POST /api/predict` with that image (as base64), the list of selected character IDs, and the current frame number.
  - The backend runs the model on the frame, computes sigmoid probabilities per class, and for each requested character runs Grad-CAM and returns an overlay image (heatmap on the frame) as base64 PNG.
5. **View results**
  The right-hand panel shows one result per selected character: **confidence** (0–100%) and a **Grad-CAM heatmap** overlay. Use the left/right arrows or the designated buttons to skip to the next character.

The frontend only talks to the backend over HTTP. No model runs in the browser. All inference and Grad-CAM are done in Python (PyTorch, OpenCV) on the server.

---


| Path                   | Description                                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `visualizer/server.py` | FastAPI app: `POST /api/model` (upload `.pth` + model type), `POST /api/predict` (frame + character IDs → predictions and Grad-CAM images).                        |
| `src/`                 | Training and evaluation: backbones (`effnet_b2`, `resnet`, `convnext_tiny`, `clip_vit_b16`), `eval/gradcam.py`, `eval/eval_single.py`, `utils.py` (model loading). |
| `visualizer/frontend/` | React + Vite app. Main UI in `visualizer/frontend/src/App.tsx`.                                                                                                    |


---

# Training Pipeline

**Before training (once):** If you want to run the training pipeline, you first need to extract frames and labels from the raw videos. Run `generate_data.py` (once) after downloading:

```bash
python generate_data.py
```

This reads `data/raw` (videos + CSV label files), extracts frames into `data/images` and labels into `data/labels`. You don't need to run this if you only use the visualizer or video prediction — the labelled data is only used for the training pipeline.

## How the Training Pipeline Works

The training pipeline is in `src/`. You run it from the **repository root** with the `src` directory on `PYTHONPATH` (e.g. `python src/train.py` or from inside `src/`). Make sure to **create and activate a virtual environment first**, then install the requirements (if not already done from above):

```bash
pip install -r requirements.txt
```

To run the training pipeline, simply execute:

```bash
python src/train.py
```

The training script uses the configuration file that is imported directly in `train.py`. If you want to use a different set of training options (for example, experiment with other model architectures or data splits), you can edit or create a configuration under `src/config/`, and then update the import line in `src/train.py` to point to your chosen config file.

## Overview of the modules

### `train.py`

  Sets device, generates an experiment group ID, builds the full list of experiment configs from the chosen `ExperimentConfig` (via `itertools.product` over all config lists). Then loops over each config and calls `run_single_experiment`. Then for each config, it sets seeds, creates the model, calls `create_dataloaders` with `episode_splits` from the config, builds loss (BCEWithLogitsLoss, optional pos_weight), optimizer (Adam), and optional scheduler (e.g. CosineAnnealingLR). For each experiment it **creates a model**, **builds dataloaders** (train/val/test) via `data_setup.create_dataloaders`, then runs **training** via `engine.train` for each single experiment.
  `**engine.train`** runs epochs (train -> optional val -> test), computes metrics, logs to **Weights & Biases**, and uses **early stopping**. When done, the script **saves the model** with `utils.save_model`.

### `data_setup.py`

   Essentially loads the labels CSV from `images_dir.parent / "labels" / "labels.csv"`.  

- If `**episode_splits` is None**: splits by fractions (train/val/test) with `train_test_split`.  
- If `**episode_splits` is given**: filters rows by episode ID (parsed from `filename`: part before the first `_`) so that train/val/test each use the listed episodes.  
 Optionally prints class distribution via `utils.analyze_class_distribution_from_df`. Builds three `MultiLabelImageDataset` instances (train, val, test) and wraps them in `DataLoader`s. If **oversampling** is enabled, uses `utils.create_weighted_sampler_from_csv` for the training dataloader. Returns `(train_dataloader, val_dataloader, test_dataloader, class_names)`.

### `dataset.py`

   PyTorch `Dataset` is a custom dataset class for multi-label image classification.  

### `engine.py`

- `**run_epoch(model, dataloader, loss_fn, device, optimizer=None)`**  
Runs one epoch over the dataloader. If `optimizer` is given, runs in train mode (forward, loss, backward, step). Otherwise runs in eval mode (no gradients). Collects all logits and targets and returns `{ "loss", "logits", "targets" }`.
- `**evaluate_metrics(logits, targets, class_names, threshold)**`  
Uses `utils.calc_metrics` to compute per-class accuracy, precision, recall, F1, and average precision, and macro mAP. Returns a dict with `per_class` and `macro` (mAP).
- `**train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, epochs, class_names, device, threshold, early_stopping_patience, scheduler)**`  
Main training loop for each single experiment from the iteration list. For each epoch, it runs train epoch, then optional val and test epochs. Steps the scheduler, computes metrics and then prints train/val/test loss and per-class test metrics. Also logs to W&B (loss curves, precision/recall per class, confusion matrices). Uses `**EarlyStopping**` on the chosen metric (mAP was selected): if there is no improvement for `early_stopping_patience` epochs, training stops.

### `early_stopping.py`

- `**EarlyStopping**`  
Tracks the best value of a metric (e.g. mAP) and the number of epochs without improvement.  
  - `**__init__(patience, metric)**`  
  `patience` = how many epochs without improvement before stopping. `metric` = key in the metrics dict (e.g. `"mAP"`).  
  - `**__call__(epoch, val_metrics)**`  
  Updates best score and `epochs_no_improve`. Returns `True` if training should stop (no improvement for `patience` epochs).

### `utils.py`

   This class defines helper methods for various services.

- `**calc_metrics(targets, logits, ...)**`  
Converts logits to probabilities (sigmoid), optionally uses per-class thresholds, and computes per-class accuracy, precision, recall, F1, and average precision, plus macro mAP.
- `**create_weighted_sampler_from_csv(df, oversample_factor)**`  
Builds a `WeightedRandomSampler` so that minority classes (e.g. rare characters) are oversampled during training. Weights are based on class frequencies in the CSV. very rare classes get an extra boost via `oversample_factor`.
- `**save_model(model, target_dir, model_name)**`  
Creates `target_dir` if needed and saves `model.state_dict()` as `target_dir/model_name` (e.g. `.pth`).
- `**get_class_names(csv_path)**`  
Reads the CSV and returns the list of column names except `filename` (the class names).
- `**analyze_class_distribution_from_df(df, label)**`  
Prints per-class positive/negative counts and imbalance ratios for the given DataFrame.
- `**load_backbone`, `load_vit_model**`  
Used at inference time (e.g. in the visualizer server) to load a saved checkpoint into a backbone and return the model.

### `config/base_config.py` and experiment configs

- `**BaseConfig**`  
Defines lists for every hyperparameter and data split: e.g. `EPISODE_SPLITS` (list of train/val/test episode dicts), `RANDOM_SEED`, `MODEL_NAME`, `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `AUGMENTATION`, `OVERSAMPLING`, `UNFREEZE_ENCODER_LAYERS`, `SCHEDULER`, etc. One experiment is one element of the Cartesian product of these lists.
- **Experiment configs** (e.g. `rq2_a4_3_combination_config.py`). Filename indicates for which research question that experiment config was used. It is a subclass of `BaseConfig` and it overrides specific fields (e.g. `TAG`, `DESCRIPTION`, `AUGMENTATION`, `OVERSAMPLING`, `OVERSAMPLE_FACTOR`, `MAX_BCE_WEIGHT`). `train.py` imports one of these as `ExperimentConfig` and uses it to build the experiment grid!

### Backbones (`src/backbone/`)

Each backbone module (e.g. `effnet_b2`, `resnet`, `convnext_tiny`, `clip_vit_b16`) defines a model class that:

- Takes `out_features` (number of classes), `pretrained`, `augmentation`, `unfreeze_last_n`, and `device`.
- Exposes `**train_transform`** and `**test_transform**` (torchvision transforms) and `**model**` (the actual `nn.Module`).  
The training script uses these transforms in `create_dataloaders` and trains `model` (or the full wrapper for CLIP) via `engine.train`.

