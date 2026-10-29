# NeuroTrace

NeuroTrace is an educational prototype that combines brain MRI classification with a verifiable processing history. It uses EfficientNet-B0 for image classification, Grad-CAM for a visual explanation, SHA-256 hashes for artifact integrity, and a local JSON ledger by default. A Hyperledger Fabric adapter is also provided for an independently configured Fabric test network.

> **Educational use only.** NeuroTrace is not a certified medical device and must not be used to make clinical decisions. Do not upload identifiable or confidential medical images. The demo stores uploaded images and results locally; users with access to a shared demo may be able to view its cases.

## What it does

For each submitted image, the application can:

1. Validate and preprocess the image.
2. Predict one of four classes: glioma, meningioma, no tumor, or pituitary tumor.
3. Apply confidence and image-quality rules, returning Accepted, Review Required, or Rejected.
4. Generate a Grad-CAM visualization when a prediction is available and not rejected.
5. Store artifacts off-chain, hash them with SHA-256, and link them in a provenance chain.
6. Verify stored files against their recorded hashes and demonstrate tamper detection.

The blockchain does not perform the classification. The default local JSON ledger is for development and demonstrations; it is not an immutable blockchain.

## Project layout

```text
BrainTumorMRI/
├── neurotrace/
│   ├── agents/             # Pipeline agents and orchestrator
│   ├── ai/                 # Model loading, prediction, Grad-CAM
│   ├── api/                # FastAPI service and web interface
│   ├── evaluation/         # Classification metrics
│   ├── hashing/            # SHA-256 and artifact records
│   ├── ledger/             # Local JSON ledger and Fabric adapter
│   ├── preprocessing/     # Image checks and preprocessing
│   ├── tests/               # Unit and API tests
│   ├── validation/         # Confidence and image-quality rules
│   ├── verification/       # Integrity and provenance checks
│   ├── evaluate.py         # Evaluate the saved model
│   └── requirements.txt
└── Projet 4.docx           # Project requirements brief
```

Local MRI datasets, model weights, generated case files, and evaluation outputs are intentionally excluded from Git.

## Requirements

- Python 3.11 recommended
- A trained EfficientNet-B0 checkpoint
- A permitted MRI dataset if you want to run evaluation

## Install and run locally

From the repository root, create a virtual environment and install the application dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r neurotrace\requirements.txt
```

Place the trained checkpoint at:

```text
neurotrace/models/efficientnet_b0/efficientnet_b0_best.pth
```

The model checkpoint and MRI images are not included in this repository. Obtain them from your authorized training/data source. The checkpoint must match the class names and image size expected by the application.

Start the API and web interface:

```powershell
cd neurotrace
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the interface or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API documentation. The app uses a local JSON ledger unless configured otherwise.

## Ledger configuration

The default backend is the local JSON ledger at `neurotrace/ledger_local/ledger.json`.

To use the Fabric adapter, first deploy the NeuroTrace chaincode to a Fabric test network and provide a working peer-wrapper script. The chaincode source and Fabric network setup are **not included in this repository version**. Set these variables in the environment where the API process runs:

```text
NEUROTRACE_LEDGER=fabric
NEUROTRACE_FABRIC_SCRIPT=<absolute WSL path to the peer-wrapper script>
NEUROTRACE_WSL_DISTRO=<your WSL distribution name>
```

The adapter submits and queries transactions through WSL. It is not suitable for a cloud host unless that host can reach a configured Fabric network and the adapter is changed for that environment.

Other optional settings:

| Variable | Purpose |
|---|---|
| `NEUROTRACE_API_KEY` | Require the `X-API-Key` header on API routes. |
| `NEUROTRACE_DEMO_MODE=1` | Enable the tamper and restore demonstration endpoints. |
| `NEUROTRACE_OFFCHAIN_DIR` | Override the directory used for case artifacts. |
| `NEUROTRACE_LEDGER_PATH` | Override the local JSON ledger file path. |
| `NEUROTRACE_UPLOAD_DIR` | Override the temporary upload directory. |

The API key setting is a shared key, not per-user authorization. Do not expose a public demo containing confidential cases.

## Evaluate the model

Prepare a test directory with one folder per checkpoint class (`glioma`, `meningioma`, `notumor`, and `pituitary`), then run:

```powershell
cd neurotrace
..\.venv\Scripts\python.exe evaluate.py --data .\Testing
```

The evaluator writes `metrics.json`, `predictions.csv`, `confusion_matrix.png`, and `threshold_table.csv` under `neurotrace/outputs/evaluation/`. The data source and license must be documented in your report; dataset files are not committed here.

## Run the tests

From the repository root:

```powershell
cd neurotrace
..\.venv\Scripts\python.exe -m pytest -q tests
```

The tests cover preprocessing, validation, the local pipeline and API, agent handling, metrics, integrity verification, and tamper detection. Run them in your environment before relying on a result.

## Public demo and privacy

The web application is an educational demonstration, not a production medical service. A public instance may expose uploaded images and case history to other visitors unless additional access controls and storage isolation are configured. Use public sample images only. The optional API key is shared by everyone who knows it, and demo mode allows case files to be deliberately modified.

## Limitations

- Model confidence is not a measure of clinical certainty.
- Grad-CAM is an interpretation aid, not proof that a prediction is correct.
- Validation thresholds are configurable and must be evaluated on appropriate data.
- The local JSON ledger is not Hyperledger Fabric and can be edited by its host.
- Fabric operation requires a separately configured test network and chaincode deployment.
