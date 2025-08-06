# Skin Lesion Classification and Localization using CNN (HAM10000)

This project implements a Convolutional Neural Network (CNN) model using PyTorch to perform **multi-task learning** on the HAM10000 dataset. It predicts:
- **Diagnosis** (`dx`) of skin lesion types (7 classes)
- **Localization** of the lesion (`localization` column in metadata)

---

## 🔬 Dataset

- **Name:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Size:** ~6GB (images)
- **Location:** Not included in this repo (you must download separately)
- **Structure:**
  - `HAM10000_images/` - Folder containing all `.jpg` images.
  - `HAM10000_metadata.csv` - CSV file containing image metadata.

---

## 🏷️ Diagnosis Classes (`dx`)

| Label   | Description                              |
|---------|------------------------------------------|
| akiec   | Actinic keratoses and intraepithelial carcinoma |
| bcc     | Basal cell carcinoma                     |
| bkl     | Benign keratosis-like lesions            |
| df      | Dermatofibroma                           |
| mel     | Melanoma                                 |
| nv      | Melanocytic nevi                         |
| vasc    | Vascular lesions                         |

---

## 🌍 Localization Labels (`localization`)

Examples from metadata:
- head
- trunk
- lower extremity
- upper extremity
- abdomen
- back
- chest
- etc.

These are converted into integer labels during training.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sohambhalerao07/skin_lision_cnn.git
cd skin_lision_cnn
