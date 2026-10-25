# EcoWaste Classifier: AI Waste Classification System

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

A computer vision system that classifies waste images into 9 categories using PyTorch CNNs and tells you how to dispose of each type. It includes a Flask backend API and a web frontend.

**Best result:** 91.79% validation accuracy (190/207 images) with EfficientNet-B0, the best of 4 architectures tested.

## Features

- Classifies waste images with trained PyTorch CNN models
- Single-model or ensemble prediction (confidence-weighted majority voting)
- Per-class confidence thresholds that reject low-confidence predictions
- Category details, decomposition/recycling instructions, and eco-friendly disposal tips
- REST API plus a web interface

## Waste categories and dataset

1,945 images across 9 categories:

| Category | Images |
|---|---|
| Battery | 228 |
| Biological | 232 |
| Clothes | 225 |
| Glass | 215 |
| Metal | 212 |
| Paper | 221 |
| Plastic | 187 |
| Shoes | 221 |
| Trash | 204 |

Cardboard is deliberately excluded because it is not in the dataset.

## Model performance

Four CNN architectures were trained on the 9-category dataset. Accuracy below is validation accuracy read directly from the saved best-epoch PyTorch checkpoints.

| Model | Val. accuracy | Epoch | Val. loss | Checkpoint |
|---|---|---|---|---|
| **EfficientNet-B0** | **91.79%** | 6 | 0.2594 | `efficientnet_b0_best.pth` |
| MobileNetV2 | 89.37% | 9 | 0.4231 | `mobilenet_v2_best.pth` |
| DenseNet121 | 86.23% | 9 | 0.3995 | `densenet121_best.pth` |
| ResNet50 | 82.37% | 9 | 0.5171 | `resnet50_best.pth` |

EfficientNet-B0 is the primary inference model for its accuracy-to-size tradeoff. Checkpoints are in `runs/classifiers/`.

## How prediction works

1. The image is classified by EfficientNet-B0 (single mode) or by several models (ensemble mode).
2. In ensemble mode, predictions are combined by confidence-weighted majority voting.
3. Per-class confidence thresholds reject low-confidence results instead of forcing a label.
4. The predicted category is returned with its disposal guidance.

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables.** Copy `.env.example` to `.env` and set `SECRET_KEY` for production deployments.

```bash
cp .env.example .env
```

**3. Start the server**

```bash
python src/gamified_api_server.py
```

**4. Open the app** at http://localhost:5000.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Classify a waste image (single or ensemble prediction) |
| `GET` | `/models` | Status of available models |
| `GET` | `/classes` | Waste class descriptions and recycling guidance |
| `GET` | `/health` | API server health |

## Limitations

- Reported accuracy is validation accuracy on 207 images. No separate held-out test result is reported, so treat 91.79% as an upper-bound estimate of real-world performance.
- The dataset is small (1,945 images) and limited to 9 categories; items outside them (for example cardboard) are not supported.
- Class sizes are roughly balanced (187 to 232 images each), but per-class metrics are not yet reported.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See the [LICENSE](LICENSE) file.
