# EcoWaste Classifier - AI Waste Classification System

This system uses trained CNN models to classify waste images into 9 categories and provides information on how to properly dispose of each type of waste. The project includes both a Flask backend API server and a web-based frontend interface.

## Features

- Classifies waste images using trained PyTorch CNN models
- Provides detailed information about each waste category
- Shows decomposition/recycling instructions
- Offers eco-friendly disposal tips
- Supports 9 waste categories:
  - Battery
  - Biological
  - Clothes
  - Glass
  - Metal
  - Paper
  - Plastic
  - Shoes
  - Trash
- Ensemble prediction logic with per-class confidence thresholds
- Cardboard category explicitly excluded (not in dataset)

## Model Performance

Four CNN architectures were trained and evaluated on the 9-category waste dataset (1,945 images). Empirical validation accuracy extracted directly from saved PyTorch checkpoints:

| Model          | Val. Accuracy | Epoch | Val. Loss | Checkpoint File |
|----------------|---------------|-------|-----------|-----------------|
| EfficientNet-B0 | 91.79%       | 6     | 0.2594    | `efficientnet_b0_best.pth` |
| MobileNetV2     | 89.37%       | 9     | 0.4231    | `mobilenet_v2_best.pth` |
| DenseNet121     | 86.23%       | 9     | 0.3995    | `densenet121_best.pth` |
| ResNet50        | 82.37%       | 9     | 0.5171    | `resnet50_best.pth` |

EfficientNet-B0 serves as the primary inference model based on its superior accuracy-to-size tradeoff. Multi-model predictions use confidence-weighted majority voting alongside per-class thresholding for low-confidence rejection.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Copy `.env.example` to `.env` and set `SECRET_KEY` for production deployments:
   ```bash
   cp .env.example .env
   ```

3. **Start the Server**
   ```bash
   python src/gamified_api_server.py
   ```
   The server will start on http://localhost:5000

4. **Access the Application**
   Open your browser and navigate to http://localhost:5000

## Dataset Information

The models were trained on a dataset containing 1,945 waste images across 9 categories:
- Battery: 228 images
- Biological: 232 images
- Clothes: 225 images
- Glass: 215 images
- Metal: 212 images
- Paper: 221 images
- Plastics: 187 images
- Shooes: 221 images
- Trash: 204 images

## API Endpoints

The Flask backend provides the following API endpoints:

- `POST /predict` - Classify a waste image using single or ensemble prediction
- `GET /models` - Get available model statuses
- `GET /classes` - Get waste class descriptions and recycling guidance
- `GET /health` - API server health status

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.