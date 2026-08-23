import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import io
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import base64
import tempfile
import traceback
import glob
from PIL import ImageFilter
import requests
import json
import hashlib
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import model functions
from all_models import get_efficientnet_b0, get_densenet121, get_mobilenet_v2, get_resnet50
from models import db, User, Post, Challenge, UserChallenge, Classification, Comment, Like, ENVIRONMENTAL_IMPACT, POINTS_SYSTEM

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'runs'))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24).hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///eco_waste.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, origins=["*"], methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])  # Enable CORS for all routes from any origin
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
# login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Serve frontend HTML
@app.route('/')
def serve_frontend():
    return app.send_static_file('EcoWarriors.html')

# Class names for the dataset
CLASS_NAMES = ['battery', 'biological', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

# Load models
def load_models():
    models = {}
    
    # Get the base directory (yolo_dataset directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifiers_dir = os.path.join(base_dir, 'runs', 'classifiers')
    main_model_path = os.path.join(os.path.dirname(base_dir), 'best_waste_classifier.pth')
    
    print(f"[DEBUG] Base directory: {base_dir}")
    print(f"[DEBUG] Classifiers directory: {classifiers_dir}")
    print(f"[DEBUG] Main model path: {main_model_path}")
    
    # Check if classifiers directory exists
    if not os.path.exists(classifiers_dir):
        print(f"[ERROR] Classifiers directory not found: {classifiers_dir}")
        # Create the directory if it doesn't exist
        os.makedirs(classifiers_dir, exist_ok=True)
    
    # Try to load main model as fallback if individual models don't exist
    main_model_checkpoint = None
    if os.path.exists(main_model_path):
        try:
            print("[DEBUG] Loading main model as fallback...")
            main_model_checkpoint = torch.load(main_model_path, map_location='cpu')
            print(f"[DEBUG] Main model checkpoint keys: {list(main_model_checkpoint.keys()) if isinstance(main_model_checkpoint, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"[ERROR] Failed to load main model: {e}")
            traceback.print_exc()
    else:
        print(f"[DEBUG] Main model file not found: {main_model_path}")
    
    try:
        # Load EfficientNet-B0 (best performing model)
        print("[DEBUG] Loading EfficientNet-B0...")
        models['efficientnet_b0'] = get_efficientnet_b0(len(CLASS_NAMES), pretrained=False)
        checkpoint_path = os.path.join(classifiers_dir, 'efficientnet_b0_best.pth')
        print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['efficientnet_b0'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['efficientnet_b0'].load_state_dict(checkpoint)
            models['efficientnet_b0'].eval()
            print("✓ EfficientNet-B0 loaded successfully")
        elif main_model_checkpoint is not None and os.path.exists(main_model_path):
            # Fallback to main model - copy it to the expected location
            print("[DEBUG] Copying main model to EfficientNet-B0 location")
            import shutil
            shutil.copy2(main_model_path, checkpoint_path)
            # Load the copied model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['efficientnet_b0'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['efficientnet_b0'].load_state_dict(checkpoint)
            models['efficientnet_b0'].eval()
            print("✓ EfficientNet-B0 loaded successfully (from main model)")
        else:
            print(f"✗ Error loading EfficientNet-B0: Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"✗ Error loading EfficientNet-B0: {e}")
        traceback.print_exc()
    
    try:
        # Load DenseNet121
        print("[DEBUG] Loading DenseNet121...")
        models['densenet121'] = get_densenet121(len(CLASS_NAMES), pretrained=False)
        checkpoint_path = os.path.join(classifiers_dir, 'densenet121_best.pth')
        print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['densenet121'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['densenet121'].load_state_dict(checkpoint)
            models['densenet121'].eval()
            print("✓ DenseNet121 loaded successfully")
        elif main_model_checkpoint is not None and os.path.exists(main_model_path):
            # Fallback to main model - copy it to the expected location
            print("[DEBUG] Copying main model to DenseNet121 location")
            import shutil
            shutil.copy2(main_model_path, checkpoint_path)
            # Load the copied model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['densenet121'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['densenet121'].load_state_dict(checkpoint)
            models['densenet121'].eval()
            print("✓ DenseNet121 loaded successfully (from main model)")
        else:
            print(f"✗ Error loading DenseNet121: Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"✗ Error loading DenseNet121: {e}")
        traceback.print_exc()
    
    try:
        # Load MobileNetV2
        print("[DEBUG] Loading MobileNetV2...")
        models['mobilenet_v2'] = get_mobilenet_v2(len(CLASS_NAMES), pretrained=False)
        checkpoint_path = os.path.join(classifiers_dir, 'mobilenet_v2_best.pth')
        print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['mobilenet_v2'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['mobilenet_v2'].load_state_dict(checkpoint)
            models['mobilenet_v2'].eval()
            print("✓ MobileNetV2 loaded successfully")
        elif main_model_checkpoint is not None and os.path.exists(main_model_path):
            # Fallback to main model - copy it to the expected location
            print("[DEBUG] Copying main model to MobileNetV2 location")
            import shutil
            shutil.copy2(main_model_path, checkpoint_path)
            # Load the copied model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['mobilenet_v2'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['mobilenet_v2'].load_state_dict(checkpoint)
            models['mobilenet_v2'].eval()
            print("✓ MobileNetV2 loaded successfully (from main model)")
        else:
            print(f"✗ Error loading MobileNetV2: Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"✗ Error loading MobileNetV2: {e}")
        traceback.print_exc()
    
    try:
        # Load ResNet50
        print("[DEBUG] Loading ResNet50...")
        models['resnet50'] = get_resnet50(len(CLASS_NAMES), pretrained=False)
        checkpoint_path = os.path.join(classifiers_dir, 'resnet50_best.pth')
        print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['resnet50'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['resnet50'].load_state_dict(checkpoint)
            models['resnet50'].eval()
            print("✓ ResNet50 loaded successfully")
        elif main_model_checkpoint is not None and os.path.exists(main_model_path):
            # Fallback to main model - copy it to the expected location
            print("[DEBUG] Copying main model to ResNet50 location")
            import shutil
            shutil.copy2(main_model_path, checkpoint_path)
            # Load the copied model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            if 'model_state_dict' in checkpoint:
                models['resnet50'].load_state_dict(checkpoint['model_state_dict'])
            else:
                models['resnet50'].load_state_dict(checkpoint)
            models['resnet50'].eval()
            print("✓ ResNet50 loaded successfully (from main model)")
        else:
            print(f"✗ Error loading ResNet50: Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"✗ Error loading ResNet50: {e}")
        traceback.print_exc()
    
    return models

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load models on startup
print("[DEBUG] Loading models...")
models = load_models()
print(f"[DEBUG] Models loaded: {list(models.keys())}")

# Enhanced waste category information with randomized tips
waste_info = {
    'battery': {
        'name': 'Battery',
        'description': 'Contains toxic heavy metals (lead, mercury, cadmium) and electrolytes that can pollute soil and water.',
        'decomposition': 'Cannot decompose naturally. Should be collected at authorized e-waste or hazardous waste centers.',
        'decay_time': 'Indefinite — batteries do not biodegrade.',
        'disposal_tips': [
            'Store used batteries in a dry container until collection.',
            'Never burn or mix with household trash.',
            'Tape the terminals before disposing to avoid short-circuiting.'
        ],
        'eco_tips': [
            'Use rechargeable batteries instead of single-use ones.',
            'Return old batteries to collection points at electronics shops.',
            'Encourage battery recycling drives in schools or communities.'
        ],
        'waste_type': 'Hazardous Waste',
        'bin_color': 'Red',
        'bin_color_code': '#dc3545',
        'icon': '🪫'
    },
    'biological': {
        'name': 'Biological',
        'description': 'Organic matter from kitchen or garden that naturally decomposes.',
        'decomposition': 'Compost in a home bin or community compost pit. Keep wet and dry waste separate for proper breakdown.',
        'decay_time': '1–3 months, depending on moisture and temperature.',
        'disposal_tips': [
            'Avoid plastic bags when disposing food waste.',
            'Compost at home for use as natural fertilizer.',
            'Mix dry leaves to balance moisture in compost.'
        ],
        'eco_tips': [
            'Start a small compost pit or vermicompost box.',
            'Reduce food waste by reusing leftovers creatively.',
            'Grow plants using your own compost.'
        ],
        'waste_type': 'Biodegradable',
        'bin_color': 'Green',
        'bin_color_code': '#28a745',
        'icon': '🍃'
    },
    'clothes': {
        'name': 'Clothes',
        'description': 'Made from fabrics like cotton, polyester, and blends; takes years to decompose.',
        'decomposition': 'Donate usable clothes; recycle worn-out textiles into rags or insulation.',
        'decay_time': '6 months (cotton) to 200 years (synthetic).',
        'disposal_tips': [
            'Donate to NGOs or clothing drives.',
            'Repurpose old clothes into bags or cleaning cloths.',
            'Avoid fast fashion; buy durable items.'
        ],
        'eco_tips': [
            'Organize clothes swaps with friends.',
            'Support sustainable clothing brands.',
            'Upcycle denim or T-shirts into crafts.'
        ],
        'waste_type': 'Non-Biodegradable',
        'bin_color': 'Blue',
        'bin_color_code': '#0f9dff',
        'icon': '👕'
    },
    'glass': {
        'name': 'Glass',
        'description': 'Made from silica and other minerals; does not biodegrade. 100% recyclable.',
        'decomposition': 'Collect clean glass; separate by color; recycle at authorized facilities.',
        'decay_time': 'Up to 1 million years if not recycled.',
        'disposal_tips': [
            'Rinse and store glass separately to avoid breakage.',
            'Never mix broken glass with food waste.',
            'Label sharp glass waste before disposal.'
        ],
        'eco_tips': [
            'Reuse glass jars for storage.',
            'Prefer glass bottles over plastic ones.',
            'Support bottle-return or refill systems.'
        ],
        'waste_type': 'Non-Biodegradable',
        'bin_color': 'Blue',
        'bin_color_code': '#0f9dff',
        'icon': '🧊'
    },
    'metal': {
        'name': 'Metal',
        'description': 'Includes aluminum, steel, and tin; recyclable and valuable as scrap.',
        'decomposition': 'Clean and send for recycling; metals can be melted and reused infinitely.',
        'decay_time': '50–500 years.',
        'disposal_tips': [
            'Crush cans to save space before recycling.',
            'Remove food residues from tins.',
            'Avoid throwing sharp metal waste loosely.'
        ],
        'eco_tips': [
            'Support businesses using recycled metals.',
            'Reuse metal containers creatively.',
            'Separate aluminum and iron waste properly.'
        ],
        'waste_type': 'Non-Biodegradable',
        'bin_color': 'Blue',
        'bin_color_code': '#0f9dff',
        'icon': '🥫'
    },
    'paper': {
        'name': 'Paper',
        'description': 'Derived from wood pulp; biodegradable and recyclable multiple times.',
        'decomposition': 'Compost or recycle clean paper; avoid mixing with plastic-coated sheets.',
        'decay_time': '2–6 weeks.',
        'disposal_tips': [
            'Keep dry and clean for recycling.',
            'Remove staples or plastic bindings.',
            'Shred confidential papers before disposal.'
        ],
        'eco_tips': [
            'Use both sides before discarding.',
            'Switch to digital notes.',
            'Choose recycled paper products.'
        ],
        'waste_type': 'Non-Biodegradable',
        'bin_color': 'Blue',
        'bin_color_code': '#0f9dff',
        'icon': '📄'
    },
    'plastic': {
        'name': 'Plastic',
        'description': 'Made from synthetic polymers; non-degradable and polluting.',
        'decomposition': 'Cannot decompose; recycle through authorized centers or reuse when possible.',
        'decay_time': '450–1000 years.',
        'disposal_tips': [
            'Clean and segregate plastic waste.',
            'Avoid single-use plastics.',
            'Drop off at plastic collection bins.'
        ],
        'eco_tips': [
            'Carry reusable bottles and bags.',
            'Replace plastic straws and cups with metal or bamboo.',
            'Support plastic-free packaging.'
        ],
        'waste_type': 'Non-Biodegradable',
        'bin_color': 'Blue',
        'bin_color_code': '#0f9dff',
        'icon': '🧴'
    },
    'shoes': {
        'name': 'Shoes',
        'description': 'Made from rubber, plastic, or fabric; slow to decompose.',
        'decomposition': 'Donate wearable pairs; recycle soles or rubber parts separately.',
        'decay_time': '30–80 years.',
        'disposal_tips': [
            'Donate old shoes to relief drives.',
            'Cut damaged shoes into smaller parts for recycling.',
            'Avoid burning as it releases toxins.'
        ],
        'eco_tips': [
            'Buy durable, repairable footwear.',
            'Participate in shoe recycling programs.',
            'Turn old shoes into planters or décor items.'
        ],
        'waste_type': 'Non-Biodegradable',
        'bin_color': 'Blue',
        'bin_color_code': '#0f9dff',
        'icon': '👟'
    },
    'trash': {
        'name': 'Trash',
        'description': 'Mixed or contaminated waste that cannot be recycled or composted.',
        'decomposition': 'Sent to landfill or incineration; minimize generation.',
        'decay_time': 'Varies from months to centuries.',
        'disposal_tips': [
            'Reduce trash by proper segregation.',
            'Avoid putting recyclables in general trash.',
            'Dispose of in sealed bags.'
        ],
        'eco_tips': [
            'Reuse before discarding.',
            'Plan purchases to reduce leftovers.',
            'Educate others about segregation.'
        ],
        'waste_type': 'Residual Waste',
        'bin_color': 'Black',
        'bin_color_code': '#000000',
        'icon': '🗑'
    }
}

def enhance_waste_info_with_web_data(category):
    """Enhance waste information by fetching data from the web"""
    # Get base information
    info = waste_info.get(category, {}).copy()  # Create a copy to avoid modifying the original
    
    # Ensure all required fields exist with default values
    required_fields = ['description', 'decomposition', 'decay_time', 'disposal_tips', 'eco_tips', 'waste_type', 'bin_color', 'bin_color_code', 'icon']
    for field in required_fields:
        if field not in info:
            if field in ['disposal_tips', 'eco_tips']:
                info[field] = ['Information not available']
            else:
                info[field] = 'Information not available'
    
    return info

# Dataset search functionality
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final 2 G')

print(f"[DEBUG] Dataset path: {DATASET_PATH}")
print(f"[DEBUG] Dataset path exists: {os.path.exists(DATASET_PATH)}")

# Category mapping to match directory names
CATEGORY_MAPPING = {
    'battery': 'Battery',
    'biological': 'Biological',
    'clothes': 'Clothes',
    'glass': 'Glass',
    'metal': 'Metal',
    'paper': 'Paper',
    'plastic': 'Plastics',  # Note the plural form in directory
    'shoes': 'Shooes',      # Match actual directory name with typo
    'trash': 'Trash'
}

def compute_image_hash(image_path):
    """Compute perceptual hash of an image for similarity comparison"""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            # Get pixel data using a safer approach
            pixels = []
            try:
                # Use numpy array approach to avoid linter issues
                img_array = np.array(img)
                pixels = img_array.flatten().tolist()
            except Exception as e:
                print(f"[DEBUG] Could not extract pixel data: {e}")
                # Fallback to manual extraction
                try:
                    pixels = []
                    width, height = img.size
                    for y in range(height):
                        for x in range(width):
                            pixels.append(img.getpixel((x, y)))
                except Exception:
                    pixels = []
            
            # Check if we have valid pixel data
            if not pixels or len(pixels) != 64:
                return None
                
            # Calculate average
            avg = sum(pixels) / len(pixels)
            # Create hash: 1 if pixel > average, 0 otherwise
            bits = "".join(['1' if pixel >= avg else '0' for pixel in pixels])
            # Convert to hexadecimal
            hex_hash = hex(int(bits, 2))[2:].zfill(16)
            return hex_hash
    except Exception as e:
        print(f"[ERROR] Failed to compute hash for {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Calculate hamming distance between two hashes"""
    if len(hash1) != len(hash2):
        return float('inf')
    
    # Convert hex to binary and count differences
    bin1 = bin(int(hash1, 16))[2:].zfill(64)
    bin2 = bin(int(hash2, 16))[2:].zfill(64)
    
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

def search_similar_images(uploaded_image_path, category, max_results=5):
    """Search for similar images in the dataset"""
    print(f"[DEBUG] Searching for similar images in category: {category}")
    
    # Get the directory for this category
    category_dir = CATEGORY_MAPPING.get(category, category)
    category_path = os.path.join(DATASET_PATH, category_dir or category)
    
    if not os.path.exists(category_path):
        print(f"[ERROR] Category directory not found: {category_path}")
        return []
    
    # Compute hash of uploaded image
    uploaded_hash = compute_image_hash(uploaded_image_path)
    if not uploaded_hash:
        return []
    
    print(f"[DEBUG] Uploaded image hash: {uploaded_hash}")
    
    # Get all images in category directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(category_path, extension)))
        image_files.extend(glob.glob(os.path.join(category_path, extension.upper())))
    
    print(f"[DEBUG] Found {len(image_files)} images in category directory")
    
    # Compute hashes and find similar images
    similar_images = []
    for image_file in image_files[:100]:  # Limit to first 100 for performance
        image_hash = compute_image_hash(image_file)
        if image_hash:
            distance = hamming_distance(uploaded_hash, image_hash)
            # Consider images with distance <= 10 as similar
            if distance <= 10:
                similar_images.append({
                    'path': image_file,
                    'filename': os.path.basename(image_file),
                    'distance': distance,
                    'similarity': max(0, 100 - (distance * 10))  # Convert to percentage
                })
    
    # Sort by similarity (highest first)
    similar_images.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"[DEBUG] Found {len(similar_images)} similar images")
    return similar_images[:max_results]

@app.route('/search_similar', methods=['POST'])
def search_similar():
    """Endpoint to search for similar images in dataset"""
    print("[DEBUG] /search_similar endpoint called")
    
    try:
        if 'image' not in request.files:
            print("[ERROR] No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("[ERROR] No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"[DEBUG] Received file: {file.filename}")
        
        # Save the uploaded image temporarily
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, file.filename or 'temp_image.jpg')
        print(f"[DEBUG] Saving image to: {image_path}")
        file.save(image_path)
        print("[DEBUG] Image saved successfully")
        
        # Get category from request or use 'all'
        category = request.form.get('category', 'all')
        print(f"[DEBUG] Search category: {category}")
        
        # Search for similar images
        similar_images = []
        if category == 'all':
            # Search in all categories
            for cat in CLASS_NAMES:
                cat_similar = search_similar_images(image_path, cat, max_results=2)
                similar_images.extend(cat_similar)
        else:
            # Search in specific category
            similar_images = search_similar_images(image_path, category)
        
        # Sort all results and take top 5
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        similar_images = similar_images[:5]
        
        # Clean up temporary file
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print("[DEBUG] Temporary image file removed")
            except Exception as e:
                print(f"[WARNING] Could not remove temporary file: {e}")
        
        # Prepare response
        response_data = {
            'similar_images': similar_images,
            'total_found': len(similar_images)
        }
        
        print(f"[DEBUG] Response data: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Exception in search_similar endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def get_ensemble_prediction(predictions, class_names):
    """Get ensemble prediction using weighted voting"""
    print(f"[DEBUG] Getting ensemble prediction from {len(predictions)} models")
    # Count votes for each class, weighted by confidence
    class_votes = {}
    total_weight = 0
    
    # Track individual model predictions for debugging
    model_predictions = {}
    
    for model_name, pred in predictions.items():
        predicted_class = class_names[pred['predicted_idx']]
        confidence = pred['confidence']
        model_predictions[model_name] = predicted_class
        
        if predicted_class not in class_votes:
            class_votes[predicted_class] = 0
        class_votes[predicted_class] += confidence
        total_weight += confidence
    
    # Debug print individual predictions
    print(f"[DEBUG] Model predictions: {model_predictions}")
    
    # Find the class with the highest weighted votes
    if class_votes:
        ensemble_class = max(class_votes.keys(), key=lambda x: class_votes[x])
        ensemble_confidence = class_votes[ensemble_class] / total_weight if total_weight > 0 else 0
        print(f"[DEBUG] Ensemble prediction: {ensemble_class} with confidence {ensemble_confidence}")
        return ensemble_class, ensemble_confidence
    
    print("[DEBUG] No ensemble prediction possible")
    return None, 0.0

def apply_confidence_rejection(ensemble_prediction, confidences, class_names):
    """Apply confidence-based rejection for uncertain predictions"""
    if ensemble_prediction[0] is None:
        print("[DEBUG] No ensemble prediction to reject")
        return None, None, "no_prediction"
    
    predicted_class, confidence = ensemble_prediction
    print(f"[DEBUG] Applying confidence rejection for {predicted_class} with confidence {confidence}")
    
    # Define confidence thresholds by class
    # Balanced thresholds for all classes
    class_thresholds = {
        'glass': 0.55,
        'plastic': 0.55,
        'paper': 0.5,
        'trash': 0.6,
        'biological': 0.45,
        'battery': 0.5,
        'clothes': 0.45,
        'metal': 0.5,
        'shoes': 0.45,
        'default': 0.5     # Default threshold for other classes
    }
    
    threshold = class_thresholds.get(predicted_class, class_thresholds['default'])
    print(f"[DEBUG] Confidence threshold for {predicted_class}: {threshold}")
    
    # Check if confidence is below threshold
    if confidence < threshold:
        print(f"[DEBUG] Confidence {confidence} below threshold {threshold}")
        return None, confidence, f"low_confidence_{predicted_class}"
    
    # Additional check: if multiple models have low confidence
    low_conf_count = sum(1 for conf in confidences.values() if conf < 0.4)
    if low_conf_count >= len(confidences) * 0.6:  # If 60% of models have low confidence
        print(f"[DEBUG] Consensus low confidence: {low_conf_count}/{len(confidences)} models have low confidence")
        return None, confidence, "consensus_low_confidence"
    
    print(f"[DEBUG] Confidence {confidence} accepted with threshold {threshold}")
    return predicted_class, confidence, None

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Enhanced prediction endpoint with detailed error logging and gamification"""
    print("[DEBUG] /predict endpoint called")
    return predict_impl(True)

@app.route('/predict_no_auth', methods=['POST'])
def predict_no_auth():
    """Prediction endpoint without authentication for testing"""
    print("[DEBUG] /predict_no_auth endpoint called")
    return predict_impl(False)

def predict_impl(use_auth=True):
    """Core prediction implementation"""
    print("[DEBUG] predict_impl called")
    
    try:
        if 'image' not in request.files:
            print("[ERROR] No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("[ERROR] No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"[DEBUG] Received file: {file.filename}")
        
        image_path = None
        try:
            # Save the uploaded image temporarily
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, file.filename or 'temp_image.jpg')
            print(f"[DEBUG] Saving image to: {image_path}")
            file.save(image_path)
            print("[DEBUG] Image saved successfully")
            
            # Load and preprocess the image
            print("[DEBUG] Loading and preprocessing image...")
            image = Image.open(image_path).convert('RGB')
            print("[DEBUG] Image opened successfully")
            
            input_tensor = transform(image)
            print(f"[DEBUG] Image transformed, tensor shape: {input_tensor.shape}")  # type: ignore
            
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension  # type: ignore
            print(f"[DEBUG] Batch dimension added, batch shape: {input_batch.shape}")
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[DEBUG] Using device: {device}")
            input_batch = input_batch.to(device)
            print("[DEBUG] Input batch moved to device")
            
            # Check if models are loaded
            if not models:
                print("[ERROR] No models loaded")
                return jsonify({'error': 'No models loaded'}), 500
            
            print(f"[DEBUG] Available models: {list(models.keys())}")
            
            # Make predictions with all models (ensemble approach)
            predictions = {}
            confidences = {}
            
            for model_name, model in models.items():
                try:
                    print(f"[DEBUG] Processing with {model_name}...")
                    model = model.to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(input_batch)
                        print(f"[DEBUG] {model_name} output shape: {output.shape}")
                        probabilities = F.softmax(output[0], dim=0)
                        print(f"[DEBUG] {model_name} probabilities shape: {probabilities.shape}")
                        
                        # Get top 3 predictions for better analysis
                        top_probs, top_indices = torch.topk(probabilities, min(3, len(CLASS_NAMES)))
                        print(f"[DEBUG] {model_name} top indices: {top_indices}")
                        print(f"[DEBUG] {model_name} top probs: {top_probs}")
                        
                        predictions[model_name] = {
                            'predicted_idx': top_indices[0].item(),
                            'confidence': float(top_probs[0].item()),
                            'top3_indices': [idx.item() for idx in top_indices],
                            'top3_probs': [float(prob) for prob in top_probs]
                        }
                        confidences[model_name] = float(top_probs[0].item())
                        print(f"[DEBUG] {model_name} prediction completed successfully")
                except Exception as e:
                    print(f"[ERROR] Error processing with {model_name}: {e}")
                    traceback.print_exc()
                    continue
            
            if not predictions:
                print("[ERROR] No predictions generated from any model")
                return jsonify({'error': 'Failed to generate predictions from models'}), 500
            
            print(f"[DEBUG] All predictions: {predictions}")
            
            # Ensemble prediction - majority vote with confidence weighting
            ensemble_prediction = get_ensemble_prediction(predictions, CLASS_NAMES)
            print(f"[DEBUG] Ensemble prediction: {ensemble_prediction}")
            
            # Debug: Print individual model predictions
            print("[DEBUG] Individual model predictions:")
            for model_name, pred in predictions.items():
                predicted_class = CLASS_NAMES[pred['predicted_idx']]
                print(f"[DEBUG]   {model_name}: {predicted_class} (confidence: {pred['confidence']:.3f})")
            
            # Confidence-based rejection
            final_prediction, final_confidence, rejection_reason = apply_confidence_rejection(
                ensemble_prediction, confidences, CLASS_NAMES
            )
            print(f"[DEBUG] Final prediction: {final_prediction}, confidence: {final_confidence}, rejection: {rejection_reason}")
            
            # Enhanced handling for biological waste
            biological_idx = None
            for i, class_name in enumerate(CLASS_NAMES):
                if class_name == 'biological':
                    biological_idx = i
                    break
            
            # Special handling for biological waste - consider multiple factors
            if biological_idx is not None:
                biological_confidences = []
                for model_name, pred in predictions.items():
                    if pred['predicted_idx'] == biological_idx:
                        biological_confidences.append(pred['confidence'])
                    # Also check if biological is in top 3
                    elif biological_idx in pred['top3_indices']:
                        idx_in_top3 = pred['top3_indices'].index(biological_idx)
                        biological_confidences.append(pred['top3_probs'][idx_in_top3])
                
                if biological_confidences:
                    avg_biological_conf = sum(biological_confidences) / len(biological_confidences)
                    # If biological has high confidence, use it
                    if avg_biological_conf > 0.6:
                        final_prediction = 'biological'
                        final_confidence = avg_biological_conf
                        rejection_reason = None
                    # If biological has moderate confidence and primary prediction is low, consider it
                    elif avg_biological_conf > 0.3 and (final_confidence is None or final_confidence < 0.5):
                        final_prediction = 'biological'
                        final_confidence = avg_biological_conf
                        rejection_reason = None
            
            # Clean up temporary file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print("[DEBUG] Temporary image file removed")
                except Exception as e:
                    print(f"[WARNING] Could not remove temporary file: {e}")
            
            # Prepare response
            response_data = {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'class_info': enhance_waste_info_with_web_data(final_prediction) if final_prediction else {},
                'predicted_class': final_prediction,
                'predictions': predictions
            }
            
            # If we have a valid prediction and authentication is enabled, update user stats
            if final_prediction and final_confidence and use_auth and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
                # Calculate CO2 saved
                co2_saved = ENVIRONMENTAL_IMPACT.get(final_prediction, 0.0)
                
                # Award points for classification
                points_earned = POINTS_SYSTEM['classification']
                
                # Save classification record
                classification = Classification()
                classification.user_id = current_user.id
                classification.waste_category = final_prediction
                classification.confidence = final_confidence
                classification.co2_saved = co2_saved
                classification.points_earned = points_earned
                classification.image_path = image_path if image_path else None
                db.session.add(classification)
                
                # Update user stats
                current_user.waste_classified += 1
                current_user.total_points += points_earned
                current_user.weekly_points += points_earned
                current_user.co2_saved += co2_saved
                
                # Check for streak update (if this is the first classification of the day)
                today = datetime.now(timezone.utc).date()
                last_active = current_user.last_active.date() if current_user.last_active else None
                
                if last_active != today:
                    # If it's been more than one day since last activity, reset streak
                    if last_active and (today - last_active).days > 1:
                        current_user.streak = 1
                    else:
                        # Continue streak
                        current_user.streak += 1
                    
                    # Award streak points
                    streak_points = POINTS_SYSTEM['daily_streak']
                    current_user.total_points += streak_points
                    current_user.weekly_points += streak_points
                    
                    # Check for streak freeze reward
                    if current_user.streak > 0 and current_user.streak % 7 == 0:
                        current_user.streak_freezes += 1
                
                current_user.last_active = datetime.now(timezone.utc)
                db.session.commit()
                
                # Add user stats to response
                response_data['user_stats'] = {
                    'streak': current_user.streak,
                    'total_points': current_user.total_points,
                    'weekly_points': current_user.weekly_points,
                    'co2_saved': current_user.co2_saved,
                    'waste_classified': current_user.waste_classified
                }
            
            if rejection_reason:
                response_data['rejection_reason'] = rejection_reason
                response_data['message'] = f"Prediction rejected due to {rejection_reason}"
            
            print(f"[DEBUG] Response data: {response_data}")
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up temporary file in case of error
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print("[DEBUG] Temporary image file removed after error")
                except Exception as cleanup_error:
                    print(f"[WARNING] Could not remove temporary file after error: {cleanup_error}")
            
            print(f"[ERROR] Exception in prediction endpoint: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[ERROR] Unexpected exception in prediction endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/models', methods=['GET'])
def get_models():
    available_models = list(models.keys())
    return jsonify({
        'models': available_models,
        'default': 'efficientnet_b0'
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': CLASS_NAMES
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'model_names': list(models.keys())
    })

# User Authentication Routes
@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validate input
        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password are required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        hashed_password = generate_password_hash(password)
        user = User()
        user.username = username
        user.email = email
        user.password_hash = hashed_password
        
        db.session.add(user)
        db.session.commit()
        
        # Log in the user
        login_user(user)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'streak': user.streak,
                'total_points': user.total_points,
                'weekly_points': user.weekly_points,
                'co2_saved': user.co2_saved,
                'waste_classified': user.waste_classified,
                'verified_posts': user.verified_posts,
                'bio': user.bio,
                'badges': user.badges,
                'streak_freezes': user.streak_freezes
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Registration error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Validate input
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Log in user
        login_user(user)
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'streak': user.streak,
                'total_points': user.total_points,
                'weekly_points': user.weekly_points,
                'co2_saved': user.co2_saved,
                'waste_classified': user.waste_classified,
                'verified_posts': user.verified_posts,
                'bio': user.bio,
                'badges': user.badges,
                'streak_freezes': user.streak_freezes
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Login error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Login failed'}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """Logout user"""
    logout_user()
    return jsonify({'message': 'Logout successful'}), 200

@app.route('/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    """Get current user profile"""
    return jsonify({
        'user': {
            'id': current_user.id,
            'username': current_user.username,
            'email': current_user.email,
            'streak': current_user.streak,
            'total_points': current_user.total_points,
            'weekly_points': current_user.weekly_points,
            'co2_saved': current_user.co2_saved,
            'waste_classified': current_user.waste_classified,
            'verified_posts': current_user.verified_posts,
            'bio': current_user.bio,
            'badges': current_user.badges,
            'streak_freezes': current_user.streak_freezes,
            'created_at': current_user.created_at.isoformat() if current_user.created_at else None,
            'last_active': current_user.last_active.isoformat() if current_user.last_active else None
        }
    }), 200

@app.route('/user/profile', methods=['PUT'])
@login_required
def update_user_profile():
    """Update user profile"""
    try:
        data = request.get_json()
        bio = data.get('bio')
        
        if bio is not None:
            current_user.bio = bio[:200]  # Limit bio to 200 characters
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'streak': current_user.streak,
                'total_points': current_user.total_points,
                'weekly_points': current_user.weekly_points,
                'co2_saved': current_user.co2_saved,
                'waste_classified': current_user.waste_classified,
                'verified_posts': current_user.verified_posts,
                'bio': current_user.bio,
                'badges': current_user.badges,
                'streak_freezes': current_user.streak_freezes
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Profile update error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Profile update failed'}), 500

# Community Hub Routes
@app.route('/posts', methods=['GET'])
def get_posts():
    """Get all posts with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        category = request.args.get('category', 'all')
        per_page = 10
        
        query = Post.query
        
        if category != 'all':
            query = query.filter_by(category=category)
        
        posts = query.order_by(Post.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        posts_data = []
        for post in posts.items:
            posts_data.append({
                'id': post.id,
                'title': post.title,
                'content': post.content,
                'category': post.category,
                'created_at': post.created_at.isoformat(),
                'likes_count': post.likes_count,
                'comments_count': post.comments_count,
                'is_verified': post.is_verified,
                'reported_count': post.reported_count,
                'author': {
                    'username': post.author.username,
                    'id': post.author.id
                }
            })
        
        return jsonify({
            'posts': posts_data,
            'pagination': {
                'page': posts.page,
                'pages': posts.pages,
                'per_page': posts.per_page,
                'total': posts.total
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Get posts error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to retrieve posts'}), 500

@app.route('/posts', methods=['POST'])
@login_required
def create_post():
    """Create a new post"""
    try:
        data = request.get_json()
        title = data.get('title')
        content = data.get('content')
        category = data.get('category')
        
        # Validate input
        if not title or not content or not category:
            return jsonify({'error': 'Title, content, and category are required'}), 400
        
        if category not in ['eco_tips', 'waste_facts', 'diy_reuse']:
            return jsonify({'error': 'Invalid category'}), 400
        
        # Create post
        post = Post()
        post.title = title[:100]  # Limit title to 100 characters
        post.content = content
        post.category = category
        post.user_id = current_user.id
        
        db.session.add(post)
        db.session.commit()
        
        return jsonify({
            'message': 'Post created successfully',
            'post': {
                'id': post.id,
                'title': post.title,
                'content': post.content,
                'category': post.category,
                'created_at': post.created_at.isoformat(),
                'likes_count': post.likes_count,
                'comments_count': post.comments_count,
                'is_verified': post.is_verified,
                'reported_count': post.reported_count,
                'author': {
                    'username': 'Unknown',
                    'id': post.user_id
                }
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Create post error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to create post'}), 500

@app.route('/posts/<int:post_id>/like', methods=['POST'])
@login_required
def like_post(post_id):
    """Like a post"""
    try:
        post = Post.query.get_or_404(post_id)
        
        # Check if user already liked this post
        existing_like = Like.query.filter_by(user_id=current_user.id, post_id=post_id).first()
        if existing_like:
            return jsonify({'error': 'You have already liked this post'}), 400
        
        # Create like
        like = Like()
        like.user_id = current_user.id
        like.post_id = post_id
        post.likes_count += 1
        
        db.session.add(like)
        db.session.commit()
        
        # Check for auto-verification (5+ likes)
        if post.likes_count >= 5 and not post.is_verified:
            post.is_verified = True
            current_user.verified_posts += 1
            current_user.total_points += POINTS_SYSTEM['verified_post']
            current_user.weekly_points += POINTS_SYSTEM['verified_post']
            db.session.commit()
            
            # Add verified badge if not already present
            if 'Verified Eco Warrior' not in current_user.badges:
                if current_user.badges:
                    current_user.badges += ',Verified Eco Warrior'
                else:
                    current_user.badges = 'Verified Eco Warrior'
                db.session.commit()
        
        return jsonify({
            'message': 'Post liked successfully',
            'likes_count': post.likes_count,
            'is_verified': post.is_verified
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Like post error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to like post'}), 500

@app.route('/posts/<int:post_id>/comment', methods=['POST'])
@login_required
def comment_on_post(post_id):
    """Comment on a post"""
    try:
        post = Post.query.get_or_404(post_id)
        data = request.get_json()
        content = data.get('content')
        
        if not content:
            return jsonify({'error': 'Content is required'}), 400
        
        # Create comment
        comment = Comment()
        comment.content = content[:500]  # Limit comment to 500 characters
        comment.user_id = current_user.id
        comment.post_id = post_id
        post.comments_count += 1
        
        db.session.add(comment)
        db.session.commit()
        
        return jsonify({
            'message': 'Comment added successfully',
            'comment': {
                'id': comment.id,
                'content': comment.content,
                'created_at': comment.created_at.isoformat(),
                'author': {
                    'username': 'Unknown',
                    'id': comment.user_id
                }
            },
            'comments_count': post.comments_count
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Comment on post error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to add comment'}), 500

@app.route('/posts/<int:post_id>/report', methods=['POST'])
@login_required
def report_post(post_id):
    """Report a post"""
    try:
        post = Post.query.get_or_404(post_id)
        post.reported_count += 1
        
        # If reported 3+ times, mark as reported
        if post.reported_count >= 3:
            post.is_reported = True
        
        db.session.commit()
        
        return jsonify({
            'message': 'Post reported successfully',
            'reported_count': post.reported_count
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Report post error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to report post'}), 500

# Leaderboard Routes
@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard (top 5 users this week)"""
    try:
        # Get top 5 users by weekly points
        top_users = User.query.order_by(User.weekly_points.desc()).limit(5).all()
        
        leaderboard = []
        for i, user in enumerate(top_users, 1):
            leaderboard.append({
                'rank': i,
                'username': user.username,
                'streak': user.streak,
                'total_points': user.total_points,
                'weekly_points': user.weekly_points,
                'verified_posts': user.verified_posts,
                'co2_saved': user.co2_saved
            })
        
        return jsonify({
            'leaderboard': leaderboard
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Get leaderboard error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to retrieve leaderboard'}), 500

# Challenges Routes
@app.route('/challenges', methods=['GET'])
def get_challenges():
    """Get all active challenges"""
    try:
        # Get challenges that are currently active
        now = datetime.now(timezone.utc)
        active_challenges = Challenge.query.filter(
            Challenge.start_date <= now,
            Challenge.end_date >= now
        ).all()
        
        challenges_data = []
        for challenge in active_challenges:
            # Get user's progress in this challenge
            user_challenge = None
            if current_user.is_authenticated:
                user_challenge = UserChallenge.query.filter_by(
                    user_id=current_user.id,
                    challenge_id=challenge.id
                ).first()
            
            challenges_data.append({
                'id': challenge.id,
                'name': challenge.name,
                'description': challenge.description,
                'start_date': challenge.start_date.isoformat(),
                'end_date': challenge.end_date.isoformat(),
                'target_count': challenge.target_count,
                'reward_points': challenge.reward_points,
                'badge_name': challenge.badge_name,
                'user_progress': {
                    'progress': user_challenge.progress if user_challenge else 0,
                    'is_completed': user_challenge.is_completed if user_challenge else False,
                    'points_earned': user_challenge.points_earned if user_challenge else 0
                } if current_user.is_authenticated else None
            })
        
        return jsonify({
            'challenges': challenges_data
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Get challenges error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to retrieve challenges'}), 500

@app.route('/challenges/<int:challenge_id>/join', methods=['POST'])
@login_required
def join_challenge(challenge_id):
    """Join a challenge"""
    try:
        challenge = Challenge.query.get_or_404(challenge_id)
        
        # Check if user already joined this challenge
        existing = UserChallenge.query.filter_by(
            user_id=current_user.id,
            challenge_id=challenge_id
        ).first()
        
        if existing:
            return jsonify({'error': 'You have already joined this challenge'}), 400
        
        # Create user challenge record
        user_challenge = UserChallenge()
        user_challenge.user_id = current_user.id
        user_challenge.challenge_id = challenge_id
        
        db.session.add(user_challenge)
        db.session.commit()
        
        return jsonify({
            'message': 'Successfully joined challenge',
            'challenge': {
                'id': challenge.id,
                'name': challenge.name,
                'description': challenge.description,
                'progress': user_challenge.progress,
                'is_completed': user_challenge.is_completed,
                'points_earned': user_challenge.points_earned
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Join challenge error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to join challenge'}), 500

@app.route('/challenges/<int:challenge_id>/progress', methods=['POST'])
@login_required
def update_challenge_progress(challenge_id):
    """Update progress in a challenge"""
    try:
        user_challenge = UserChallenge.query.filter_by(
            user_id=current_user.id,
            challenge_id=challenge_id
        ).first_or_404()
        
        challenge = user_challenge.challenge
        
        # Update progress
        user_challenge.progress += 1
        
        # Check if challenge is completed
        if user_challenge.progress >= challenge.target_count and not user_challenge.is_completed:
            user_challenge.is_completed = True
            user_challenge.completed_at = datetime.now(timezone.utc)
            user_challenge.points_earned = challenge.reward_points
            
            # Award points to user
            current_user.total_points += challenge.reward_points
            current_user.weekly_points += challenge.reward_points
            
            # Award badge if specified
            if challenge.badge_name:
                if current_user.badges:
                    current_user.badges += f',{challenge.badge_name}'
                else:
                    current_user.badges = challenge.badge_name
        
        db.session.commit()
        
        return jsonify({
            'message': 'Progress updated successfully',
            'progress': user_challenge.progress,
            'is_completed': user_challenge.is_completed,
            'points_earned': user_challenge.points_earned
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Update challenge progress error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to update progress'}), 500

# Impact Tracker Routes
@app.route('/impact', methods=['GET'])
@login_required
def get_impact_data():
    """Get user's environmental impact data"""
    try:
        # Get user's impact data
        user_impact = {
            'co2_saved': current_user.co2_saved,
            'waste_classified': current_user.waste_classified,
            'streak': current_user.streak,
            'verified_posts': current_user.verified_posts
        }
        
        # Get community impact data (top 5 users by CO2 saved)
        top_contributors = User.query.order_by(User.co2_saved.desc()).limit(5).all()
        community_impact = {
            'total_co2_saved': sum(user.co2_saved for user in User.query.all()),
            'total_waste_classified': sum(user.waste_classified for user in User.query.all()),
            'top_contributors': [
                {
                    'username': user.username,
                    'co2_saved': user.co2_saved
                } for user in top_contributors
            ]
        }
        
        return jsonify({
            'user_impact': user_impact,
            'community_impact': community_impact
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Get impact data error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to retrieve impact data'}), 500

@app.route('/community/stats', methods=['GET'])
def get_community_stats():
    """Get community-wide statistics"""
    try:
        # Get total statistics across all users
        total_users = User.query.count()
        total_classifications = db.session.query(db.func.sum(User.waste_classified)).scalar() or 0
        total_co2_saved = db.session.query(db.func.sum(User.co2_saved)).scalar() or 0.0
        
        # Get this week's start date (Monday)
        today = datetime.now(timezone.utc).date()
        week_start = today - timedelta(days=today.weekday())
        week_start_dt = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc)
        
        # Get weekly classifications
        weekly_classifications = db.session.query(
            db.func.sum(User.waste_classified)
        ).filter(User.last_active >= week_start_dt).scalar() or 0
        
        # Get top contributors
        top_contributors = User.query.order_by(User.co2_saved.desc()).limit(10).all()
        
        return jsonify({
            'total_users': total_users,
            'total_classifications': total_classifications,
            'total_co2_saved': round(total_co2_saved, 2),
            'weekly_classifications': weekly_classifications,
            'top_contributors': [
                {
                    'username': user.username,
                    'co2_saved': round(user.co2_saved, 2),
                    'waste_classified': user.waste_classified,
                    'streak': user.streak
                } for user in top_contributors
            ]
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Get community stats error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to retrieve community stats'}), 500

# Initialize database
# Replace deprecated @app.before_first_request with app context
with app.app_context():
    db.create_all()
    
    # Create sample challenges if none exist
    if Challenge.query.count() == 0:
        sample_challenges = [
            Challenge(),
            Challenge(),
            Challenge()
        ]
        
        # Set attributes for first challenge
        sample_challenges[0].name = "Plastic-Free July"
        sample_challenges[0].description = "Classify 20 plastic items this month to earn bonus points and the Plastic-Free Champion badge!"
        sample_challenges[0].start_date = datetime.now(timezone.utc)
        sample_challenges[0].end_date = datetime.now(timezone.utc) + timedelta(days=30)
        sample_challenges[0].target_count = 20
        sample_challenges[0].reward_points = 50
        sample_challenges[0].badge_name = "Plastic-Free Champion"
        
        # Set attributes for second challenge
        sample_challenges[1].name = "Compost Challenge Week"
        sample_challenges[1].description = "Classify 10 biological waste items this week to earn bonus points and the Compost Hero badge!"
        sample_challenges[1].start_date = datetime.now(timezone.utc)
        sample_challenges[1].end_date = datetime.now(timezone.utc) + timedelta(days=7)
        sample_challenges[1].target_count = 10
        sample_challenges[1].reward_points = 30
        sample_challenges[1].badge_name = "Compost Hero"
        
        # Set attributes for third challenge
        sample_challenges[2].name = "Recycle Marathon Month"
        sample_challenges[2].description = "Classify 30 recyclable items (paper, glass, metal) this month to earn bonus points and the Recycling Master badge!"
        sample_challenges[2].start_date = datetime.now(timezone.utc)
        sample_challenges[2].end_date = datetime.now(timezone.utc) + timedelta(days=30)
        sample_challenges[2].target_count = 30
        sample_challenges[2].reward_points = 75
        sample_challenges[2].badge_name = "Recycling Master"
        
        for challenge in sample_challenges:
            db.session.add(challenge)
        
        db.session.commit()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
