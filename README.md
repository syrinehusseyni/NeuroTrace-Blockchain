
# NeuroTrace

**AI-Based Brain Tumor Classification and Secure Provenance Using Blockchain**

NeuroTrace is an engineering project exploring brain tumor classification from MRI images using deep learning, explainability, and secure provenance tracking.

## Current Progress

- [x] Train an EfficientNet-B0 classification model in Google Colab
- [x] Run local MRI inference using PyTorch
- [x] Generate Grad-CAM heatmaps and visualizations
- [x] Calculate SHA-256 hashes for MRI images and model checkpoints
- [x] Generate a local JSON provenance record
- [ ] Test file-tampering detection
- [ ] Integrate Hyperledger Fabric
- [ ] Develop a backend API
- [ ] Build a user interface and case history
- [ ] Explore agentic orchestration

## Technologies

- Python
- PyTorch and Torchvision
- EfficientNet-B0
- Grad-CAM
- SHA-256
- JSON
- Hyperledger Fabric (planned)

## Project Structure

```text
BrainTumorMRI/
├── src/
│   ├── inference.py
│   ├── gradcam.py
│   └── provenance.py
├── Images/             # Local test images; not tracked
├── models/             # Model checkpoints; not tracked
├── outputs/            # Generated results; not tracked
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place a test MRI image in the local `Images/` directory and the trained checkpoint at:

```text
models/efficientnet_b0/efficientnet_b0_best.pth
```

Run inference:

```powershell
python src/inference.py
```

Run Grad-CAM:

```powershell
python src/gradcam.py
```

Generate a provenance record:

```powershell
python src/provenance.py
```

## Disclaimer

This is an educational and research prototype, not a medical diagnostic system. Model outputs and Grad-CAM visualizations must not be used as a substitute for professional medical evaluation.
