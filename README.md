# Novel Skin Disease Classification Using a GA-CNN Hybrid Architecture

A Genetic Algorithm searches CNN architecture and hyperparameters; the winning
configuration is trained to convergence and served through a Streamlit app that
classifies dermoscopic skin lesion images into the 7 HAM10000 classes.

| Code | Diagnosis | Risk |
| --- | --- | --- |
| MEL | Melanoma | Malignant |
| BCC | Basal cell carcinoma | Malignant |
| AKIEC | Actinic keratosis / Bowen's disease | Pre-cancerous |
| BKL | Benign keratosis-like lesion | Benign |
| DF | Dermatofibroma | Benign |
| NV | Melanocytic nevus | Benign |
| VASC | Vascular lesion | Benign |

## Files

| File | Purpose |
| --- | --- |
| `preprocess.py` | DullRazor hair removal, CLAHE, Dataset/DataLoader, class weights |
| `model.py` | Configurable CNN backbones (custom / ResNet18 / EfficientNet-B0) + optimizers |
| `ga_optimizer.py` | PyGAD genetic search, fitness = validation macro F1 after warm-up epochs |
| `train.py` | Full training, EarlyStopping, ReduceLROnPlateau, metrics and confusion matrix |
| `app.py` | Streamlit upload UI with diagnosis, confidence, risk tag, top-3 chart |
| `dataset_preparation.ipynb` | Exploratory Data Analysis (EDA) and demographic Linear Regression |
## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset folder

Download HAM10000 (Kaggle: `kmader/skin-cancer-mnist-ham10000`) and arrange it as:

```
data/
  HAM10000_metadata.csv
  images/                 # all images from part_1 and part_2 in one folder
    ISIC_0024306.jpg
    ...
```

Sanity check the pipeline and class balance:

```bash
python preprocess.py --data-dir data
```

## 3. Run the genetic search

```bash
python ga_optimizer.py --data-dir data --generations 5 --population 8 --warmup-epochs 3
```

Each candidate is trained for a few epochs on a 25% stratified subset
(`--subset-fraction`) and scored by validation macro F1. Outputs
`best_model_config.json` and `ga_search_history.json`.

Chromosome genes: backbone, filter/dense width, dropout, learning rate,
optimizer, batch size. Tournament selection, two-point crossover, random
mutation, 1 elite kept per generation.

## 4. Train the winning architecture

```bash
python train.py --data-dir data --config best_model_config.json --epochs 30
# class-weighted focal loss instead of weighted cross-entropy:
python train.py --loss focal
```

Outputs `best_skin_model.pth`, `training_report.txt`, `confusion_matrix.png`.

## 4b. Quick CPU Testing (For Evaluation)
If you are testing this project locally on a standard CPU without a dedicated CUDA GPU, you can use these minimized flags to verify the end-to-end pipeline completes in under 20 minutes:

```bash
python ga_optimizer.py --generations 1 --population 2 --warmup-epochs 1 --subset-fraction 0.05 --no-clean
python train.py --epochs 1

## 5. Launch the app

```bash
streamlit run app.py
```

Upload a JPG/PNG lesion image to see the predicted disease, confidence,
colour-coded risk tag and the top-3 probability breakdown.

## Notes

- CPU works but is slow; a CUDA GPU is strongly recommended. Reduce
  `--population`, `--generations` or `--subset-fraction` on CPU.
- Add `--no-clean` to the GA run to skip DullRazor/CLAHE during search only.
- Research and educational use only — not a medical device.
