# Brain Tumor Segmentation & Classification (U-Net + EfficientNet/ResNet)

MRI -> preprocessing -> **U-Net** (tumor mask) -> crop tumor -> **EfficientNet-B0 / ResNet-18** (tumor type) -> web result.

## 1. Setup (Python 3.10 - 3.12)
```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
Pinned versions are in `requirements.txt`. A GPU is optional; CPU works for small datasets.

## 2. Put your dataset in `data/`
```
data/segmentation/images/img001.png      <- MRI slices
data/segmentation/masks/img001.png       <- binary mask, SAME file name (or img001_mask.png)
data/classification/glioma/*.jpg         <- one folder per tumor type
data/classification/meningioma/*.jpg
data/classification/pituitary/*.jpg
```
- Folder names (`notumor`, `normal`, ...) are skipped automatically for tumor-TYPE classification.
- Masks: any non-zero pixel = tumor. Images may be png/jpg/bmp/tif.
- The two folders can come from different datasets (one with masks, one with class labels).
- Splits are automatic: 70% train / 15% val / 15% test (fixed seed).

**No dataset yet?** `python make_synthetic_data.py` creates fake images to smoke-test the code
(delete the generated files in `data/` before using real data).

## 3. Train (in this order)
```
python train_segmentation.py --epochs 30
python train_classification.py --arch efficientnet_b0 --epochs 20     # or --arch resnet18
```
Outputs: `checkpoints/unet.pt`, `checkpoints/classifier.pt`, and plots in `outputs/`
(`seg_curves.png`, `seg_samples.png`, `cls_curves.png`, `cls_confusion_matrix.png`).
Test Dice/IoU/accuracy are printed at the end. The classifier downloads pretrained ImageNet
weights on first run (internet needed once).

## 4. Run the web app
```
uvicorn backend.app:app --reload
```
Open http://127.0.0.1:8000 -> upload an MRI -> see mask, overlay, tumor type and probabilities.

## Syllabus concepts -> where in the code
| Concept | Where |
|---|---|
| Neural networks | `models.py` (U-Net, EfficientNet/ResNet) |
| Gradient descent | `train_*.py` (`loss.backward()` + `optimizer.step()`, Adam/AdamW) |
| Train/val/test split | `datasets.split_indices` |
| Preprocessing / feature engineering | `utils.normalize_mri`, resize, tumor crop, augmentation in `datasets.py` |
| Overfitting control | Dropout, augmentation, weight decay, LR scheduler, early stopping; compare train vs val curves |
| Classification / decision boundaries | final Linear + softmax layer; confusion matrix plot |

## Notes
- Educational demo only, not a medical device.
- If tumor area < 0.2% of the image the app reports "No tumor detected" (`config.MIN_AREA_FRAC`).
