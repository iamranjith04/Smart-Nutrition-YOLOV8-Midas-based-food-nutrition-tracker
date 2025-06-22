from flask import Flask, render_template, session, redirect, url_for, request, jsonify, Response
from functools import wraps, lru_cache
import pymongo
from passlib.hash import pbkdf2_sha256
from datetime import datetime
import math
from bson.objectid import ObjectId
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import time
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import ssl
from concurrent.futures import ThreadPoolExecutor
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'


try:
    client = pymongo.MongoClient("mongodb+srv://<user>:<password>.vd87zs7.mongodb.net/?retryWrites=true&w=majority&appName=<clusterNo>")
    client.server_info()
    db = client['user_login_system']
    print("Connected to MongoDB Atlas!")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    try:
        client = pymongo.MongoClient('localhost', 27017)
        client.server_info()
        db = client.user_login_system
        print("Connected to Local MongoDB!")
    except Exception as e:
        print(f"Error connecting to local MongoDB: {e}")
        print("Please ensure MongoDB is running and accessible")

os.makedirs('tempupload', exist_ok=True)

@app.route('/tempupload/<filename>')
def serve_temp_file(filename):
    return send_from_directory('tempupload', filename)


credit_model = YOLO('Models/creditbest.pt')
food_model = YOLO('Models/foodbest.pt')
food_model.fuse()


if torch.cuda.is_available():
    food_model = food_model.half().cuda()

nutrition_df = pd.read_csv('full_food_nutrition_data.csv')
CREDIT_CARD_WIDTH = 8.56  # cm
CREDIT_CARD_HEIGHT = 5.398  # cm
CREDIT_CARD_AREA = CREDIT_CARD_WIDTH * CREDIT_CARD_HEIGHT  # cm^2
TYPICAL_FOOD_HEIGHT_MIN = 0.5  # cm
TYPICAL_FOOD_HEIGHT_MAX = 5.0  # cm

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return float(obj.item())
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_midas_model():
    midas_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    if torch.cuda.is_available():
        model = model.half().cuda()
    model.eval()
    return model, midas_transforms

midas_model, midas_transforms = load_midas_model()

@lru_cache(maxsize=100)
def get_nutrition_info(food_item, volume_cm3):
    food_data = nutrition_df[nutrition_df['Food Item'].str.lower() == food_item.lower()]
    
    if food_data.empty:
        return None
    
    food_data = food_data.iloc[0]
    
    if not pd.isna(food_data['Serving Size']):
        return {
            'weight_g': float(food_data['Serving Size'].split(' ')[0]),
            'calories': float(food_data['Calories']),
            'protein_g': float(food_data['Protein (g)']),
            'carbs_g': float(food_data['Carbs (g)']),
            'fats_g': float(food_data['Fats (g)']),
            'fiber_g': float(food_data['Fiber (g)']),
            'is_serving': True
        }
    else:
        density = food_data['Density (g/cm³)']
        weight_g = volume_cm3 * density
        
        return {
            'weight_g': float(round(weight_g, 2)),
            'calories': float(round(weight_g * food_data['Calories'] / 100, 2)),
            'protein_g': float(round(weight_g * food_data['Protein (g)'] / 100, 2)),
            'carbs_g': float(round(weight_g * food_data['Carbs (g)'] / 100, 2)),
            'fats_g': float(round(weight_g * food_data['Fats (g)'] / 100, 2)),
            'fiber_g': float(round(weight_g * food_data['Fiber (g)'] / 100, 2)),
            'is_serving': False
        }

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session and session['logged_in']:
            return f(*args, **kwargs)
        return redirect(url_for('login'))
    return wrap

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        if not all([name, email, password]):
            return render_template('signup.html', error='All fields are required')
        if db.users.find_one({'email': email}):
            return render_template('signup.html', error='Account already exists. Please login.')
        user = {
            '_id': email,
            'name': name,
            'email': email,
            'password': pbkdf2_sha256.encrypt(password),
            'created_at': datetime.utcnow()
        }
        db.users.insert_one(user)
        session['logged_in'] = True
        session['user'] = {'name': name, 'email': email}
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = db.users.find_one({'email': email})
        if user and pbkdf2_sha256.verify(password, user['password']):
            session['logged_in'] = True
            session['user'] = {'name': user['name'], 'email': user['email']}
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/home')
@login_required
def home():
    user = session.get('user')
    return render_template('home.html', user=user)

@app.route('/get_nutrition_data')
@login_required
def get_nutrition_data():
    email = session['user']['email']
    profile = db.profiles.find_one({'user_email': email})
    
    if profile and '_id' in profile:
        profile['_id'] = str(profile['_id'])
    
    daily_totals = db.daily_nutrition.find_one({
        'user_email': email,
        'date': datetime.utcnow().strftime('%Y-%m-%d')
    })
    
    if daily_totals and '_id' in daily_totals:
        daily_totals['_id'] = str(daily_totals['_id'])
    else:
        daily_totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
    
    return jsonify({
        'profile': profile,
        'daily_totals': daily_totals
    })

@app.route('/save_profile', methods=['POST'])
@login_required
def save_profile():
    try:
        data = request.get_json()
        email = session['user']['email']
      
        required_fields = {
            'age': int(data.get('age', 25)),
            'sex': data.get('sex', 'male'),
            'weight': float(data.get('weight', 70)),
            'height': float(data.get('height', 170)),
            'activity_level': data.get('activity_level', 'moderate'),
            'goal': data.get('goal', 'maintenance'),
            'health_conditions': data.get('health_conditions', []),
           
            'occupation_type': data.get('occupation_type', ''),
            'exercise_duration': float(data.get('exercise_duration', 0)),
            'exercise_intensity': data.get('exercise_intensity', 'low'),
            'daily_steps': int(data.get('daily_steps', 5000)),
            'weekly_exercise': int(data.get('weekly_exercise', 0))
        }
        
        targets = calculate_nutrition_targets(required_fields)
        
        profile = {
            'user_email': email,
            'age': required_fields['age'],
            'sex': required_fields['sex'],
            'weight': required_fields['weight'],
            'height': required_fields['height'],
            'activity_level': targets.get('activity_level', required_fields['activity_level']),
            'goal': required_fields['goal'],
            'health_conditions': required_fields['health_conditions'],
            **targets,
            'last_updated': datetime.utcnow()
        }
        
        db.profiles.update_one(
            {'user_email': email},
            {'$set': profile},
            upsert=True
        )
        
        profile_for_response = db.profiles.find_one({'user_email': email})
        if profile_for_response and '_id' in profile_for_response:
            profile_for_response['_id'] = str(profile_for_response['_id'])
        
        return jsonify({'success': True, 'profile': profile_for_response})
    except Exception as e:
        print(f"Error in save_profile: {e}")
        import traceback
        traceback.print_exc() 
        return jsonify({'success': False, 'error': str(e)}), 400
    
@app.route('/add_nutrition', methods=['POST'])
@login_required
def add_nutrition():
    try:
        data = request.get_json()
        email = session['user']['email']
        today = datetime.utcnow().strftime('%Y-%m-%d')

        calories = float(data.get('calories', 0) or 0)
        protein = float(data.get('protein', 0) or 0)
        carbs = float(data.get('carbs', 0) or 0)
        fats = float(data.get('fats', 0) or 0)
        food_name = data.get('food_name', 'Unnamed food')

        if any(math.isnan(x) or math.isinf(x) for x in [calories, protein, carbs, fats]):
            return jsonify({'success': False, 'error': 'Invalid nutrition values'}), 400

        result = db.daily_nutrition.update_one(
            {'user_email': email, 'date': today},
            {'$inc': {
                'calories': calories,
                'protein': protein,
                'carbs': carbs,
                'fats': fats
            }},
            upsert=True
        )
        
        food_entry = {
            'user_email': email,
            'date': today,
            'timestamp': datetime.utcnow(),
            'calories': calories,
            'protein': protein,
            'carbs': carbs,
            'fats': fats,
            'food_name': food_name
        }
        
        entry_result = db.food_entries.insert_one(food_entry)
       
        updated_totals = db.daily_nutrition.find_one({'user_email': email, 'date': today})
        if updated_totals:
            updated_totals['_id'] = str(updated_totals['_id'])
        
        return jsonify({
            'success': True, 
            'message': 'Nutrition data added successfully',
            'entry_id': str(entry_result.inserted_id),
            'updated_totals': updated_totals
        })
    except ValueError as e:
        print(f"Value error in add_nutrition: {e}")
        return jsonify({'success': False, 'error': 'Invalid nutrition data format'}), 400
    except Exception as e:
        print(f"Error in add_nutrition: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/process_frame', methods=['POST'])
def process_frame():
    start_time = time.time()
    focal_length_px = request.form.get('focal_length', type=float)
    if not focal_length_px:
        return jsonify({'error': 'Focal length is required'}), 400
    
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    file = request.files['frame']
    if file.filename == '':
        return jsonify({'error': 'No frame provided'}), 400
    
    frame_data = file.read()
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Failed to decode frame'}), 400
    
    # Store original frame for saving
    original_frame = frame.copy()
    
    # Resize for detection but keep aspect ratio
    original_height, original_width = frame.shape[:2]
    target_width = 640
    target_height = int(original_height * (target_width / original_width))
    small_frame = cv2.resize(frame, (target_width, target_height))
    
    # Run detection on small frame
    results = credit_model(small_frame, imgsz=640, conf=0.4, verbose=False)

    response_data = {'in_range': False, 'distance': None, 'saved_image': None}
 
    if len(results[0].boxes.data) > 0:
        boxes = results[0].boxes.data
        confidences = boxes[:, 4]
        best_idx = torch.argmax(confidences)
        box = boxes[best_idx]

        # Scale back to original frame coordinates
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        x1, y1, x2, y2 = (box[:4] * torch.tensor([scale_x, scale_y, scale_x, scale_y])).int().tolist()
        
        card_width_px = x2 - x1
        distance_cm = (focal_length_px * CREDIT_CARD_WIDTH) / card_width_px
        response_data['distance'] = round(distance_cm, 2)

        # Create display frame with border and annotations
        frame_with_border = cv2.copyMakeBorder(
            frame, 10, 10, 10, 10, 
            cv2.BORDER_CONSTANT, 
            value=(0, 255, 0) if 25.0 <= distance_cm <= 30.0 else (0, 0, 255)
        )
        
        cv2.rectangle(frame_with_border, (x1+10, y1+10), (x2+10, y2+10), (0, 255, 0), 2)
        distance_text = f"Distance: {distance_cm:.2f} cm"
        cv2.putText(frame_with_border, distance_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        conf_text = f"Conf: {box[4]:.2f}"
        cv2.putText(frame_with_border, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save image when in range - use ORIGINAL frame, not the one with borders
        if 25.0 <= distance_cm <= 30.0:
            timestamp = int(time.time())
            saved_image_path = os.path.join('tempupload', f'food_image_{timestamp}.jpg')
            
            # CRITICAL FIX: Save the original frame, not the bordered one
            success = cv2.imwrite(saved_image_path, original_frame)
            
            if success:
                response_data['in_range'] = True
                response_data['saved_image'] = os.path.basename(saved_image_path)
                
                # Also save detection metadata for later use
                metadata = {
                    'focal_length': focal_length_px,
                    'distance': distance_cm,
                    'card_bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(box[4])
                }
                metadata_path = os.path.join('tempupload', f'metadata_{timestamp}.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                
                cv2.putText(frame_with_border, "Image captured! Processing...", (50, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                print(f"Failed to save image to {saved_image_path}")
    else:
        frame_with_border = cv2.copyMakeBorder(
            frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        cv2.putText(frame_with_border, "No card detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Encode the display frame for streaming
    frame_size = frame_with_border.shape[0] * frame_with_border.shape[1]
    jpeg_quality = 90 if frame_size < 1000000 else 80
    _, buffer = cv2.imencode('.jpg', frame_with_border, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    frame_bytes = buffer.tobytes()
    
    processing_time = time.time() - start_time
    
    return Response(
        response=frame_bytes,
        mimetype='image/jpeg',
        headers={
            'X-In-Range': str(response_data['in_range']),
            'X-Distance': str(response_data['distance']) if response_data['distance'] else '',
            'X-Saved-Image': response_data['saved_image'] if response_data['saved_image'] else '',
            'X-Processing-Time': f"{processing_time:.2f}",
            'X-Detected': 'true' if len(results[0].boxes.data) > 0 else 'false'
        }
    )

def process_food_item(box, image, real_depth_map, per_pixel_area, background_depth):
    x1, y1, x2, y2, conf, cls = box
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    roi_margin = int(min(x2-x1, y2-y1) * 0.1)
    x1_roi = max(0, x1 + roi_margin)
    y1_roi = max(0, y1 + roi_margin)
    x2_roi = min(image.shape[1], x2 - roi_margin)
    y2_roi = min(image.shape[0], y2 - roi_margin)

    food_area_px = (x2_roi - x1_roi) * (y2_roi - y1_roi)
    food_area_cm2 = food_area_px * per_pixel_area
    food_depths = real_depth_map[y1_roi:y2_roi, x1_roi:x2_roi].flatten()
    food_depths = food_depths[~np.isnan(food_depths)]
    
    if len(food_depths) == 0:
        return None
    
    food_depths = np.sort(food_depths)
    lower_idx = int(len(food_depths) * 0.1)
    upper_idx = int(len(food_depths) * 0.9)
    food_depths = food_depths[lower_idx:upper_idx]
    min_food_depth = np.min(food_depths)
    height_cm = max(0, background_depth - min_food_depth)
    height_cm = min(max(height_cm, TYPICAL_FOOD_HEIGHT_MIN), TYPICAL_FOOD_HEIGHT_MAX)
    volume_cm3 = food_area_cm2 * height_cm
    class_id = int(cls)
    class_name = food_model.names[class_id]

    nutrition_info = get_nutrition_info(class_name, volume_cm3)
    
    if nutrition_info:
        return {
            'id': int(box[4].item()),  
            'class': class_name,
            'confidence': float(conf),
            'area_cm2': round(food_area_cm2, 2),
            'volume_cm3': round(volume_cm3, 2),
            'weight_g': nutrition_info['weight_g'],
            'calories': nutrition_info['calories'],
            'protein_g': nutrition_info['protein_g'],
            'carbs_g': nutrition_info['carbs_g'],
            'fats_g': nutrition_info['fats_g'],
            'fiber_g': nutrition_info['fiber_g'],
            'is_serving': nutrition_info['is_serving'],
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        }
    else:
        return {
            'id': int(box[4].item()),
            'class': class_name,
            'confidence': float(conf),
            'area_cm2': round(food_area_cm2, 2),
            'volume_cm3': round(volume_cm3, 2),
            'weight_g': None,
            'calories': None,
            'protein_g': None,
            'carbs_g': None,
            'fats_g': None,
            'fiber_g': None,
            'is_serving': None,
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        }

@app.route('/focal_length')
@login_required
def focal_length():
    return render_template('focal_length.html')

@app.route('/calculate_focal_length', methods=['POST'])
def calculate_focal_length():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        frame_data = image_file.read()
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400

        results = credit_model(frame, imgsz=640, conf=0.4, verbose=False)
        
        if len(results[0].boxes.data) == 0:
            return jsonify({'error': 'No credit card detected. Please ensure the card is clearly visible and try again.'}), 400

        boxes = results[0].boxes.data
        confidences = boxes[:, 4]
        best_idx = torch.argmax(confidences)
        box = boxes[best_idx]
        
        x1, y1, x2, y2 = box[:4].int().tolist()
        width_px = x2 - x1
        height_px = y2 - y1

        focal_length_px = (width_px * 30) / 8.56

        return jsonify({
            'success': True,
            'focal_length_px': round(float(focal_length_px), 2),
            'confidence': round(float(box[4]), 2),
            'bounding_box': [float(x1), float(y1), float(x2), float(y2)],
            'card_dimensions_px': {
                'width': float(width_px),
                'height': float(height_px)
            },
            'image_size': [frame.shape[1], frame.shape[0]]
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        image_filename = data.get('image')
        if not image_filename:
            return jsonify({'error': 'No image filename provided'}), 400
            
        image_path = os.path.join('tempupload', image_filename)
        if not os.path.exists(image_path):
            return jsonify({
                'error': f'Image not found at path: {image_path}', 
                'should_retry': False
            }), 404
        
        focal_length_px = float(data.get('focal_length', 0))
        if focal_length_px <= 0:
            return jsonify({'error': 'Invalid focal length'}), 400
       
        # Read and validate image
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            return jsonify({
                'error': 'Failed to read image or image is corrupted', 
                'path': image_path,
                'should_retry': True
            }), 400

        print(f"Processing image: {image_path}, Shape: {image.shape}")
        
        # Try different detection strategies
        credit_detected = False
        card_box = None
        
        # Strategy 1: Try with original image first
        try:
            credit_results = credit_model(image, imgsz=640, conf=0.3, verbose=False)  # Lower confidence
            if len(credit_results[0].boxes.data) > 0:
                credit_detected = True
                card_box = credit_results[0].boxes.data[0]
                print(f"Credit card detected in original image with confidence: {card_box[4]:.3f}")
        except Exception as e:
            print(f"Error in original image detection: {e}")
        
        # Strategy 2: If not detected, try with resized image
        if not credit_detected:
            try:
                # Try multiple resize ratios
                for scale in [0.8, 1.2, 0.6, 1.5]:
                    h, w = image.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized_image = cv2.resize(image, (new_w, new_h))
                    
                    credit_results = credit_model(resized_image, imgsz=640, conf=0.25, verbose=False)
                    if len(credit_results[0].boxes.data) > 0:
                        # Scale back the coordinates
                        card_box = credit_results[0].boxes.data[0]
                        card_box[:4] = card_box[:4] / scale  # Scale back coordinates
                        credit_detected = True
                        print(f"Credit card detected in {scale}x scaled image with confidence: {card_box[4]:.3f}")
                        break
            except Exception as e:
                print(f"Error in scaled image detection: {e}")
        
        # Strategy 3: Check if we have saved metadata from the frame processing
        if not credit_detected:
            try:
                timestamp = image_filename.replace('food_image_', '').replace('.jpg', '')
                metadata_path = os.path.join('tempupload', f'metadata_{timestamp}.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Use the saved bounding box from frame processing
                    x1, y1, x2, y2 = metadata['card_bbox']
                    confidence = metadata['confidence']
                    
                    # Create a fake box tensor for consistency
                    card_box = torch.tensor([x1, y1, x2, y2, confidence, 0])  # class 0 for credit card
                    credit_detected = True
                    print(f"Using saved metadata for credit card detection")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        if not credit_detected:
            # Save debug image
            debug_path = os.path.join('tempupload', f'debug_no_card_{image_filename}')
            cv2.imwrite(debug_path, image)
            return jsonify({
                'error': 'Credit card not detected in saved image. This might be due to compression or lighting changes.',
                'debug_image': os.path.basename(debug_path),
                'should_retry': True,
                'suggestions': [
                    'Ensure the credit card is clearly visible',
                    'Check lighting conditions',
                    'Make sure the card is not too close to edges',
                    'Try recalibrating focal length'
                ]
            }), 400
        
        # Process the detected card
        x1, y1, x2, y2 = card_box[:4].int().tolist()
        
        # Validate bounding box
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return jsonify({'error': 'Invalid card bounding box detected'}), 400
        
        card_width_px = x2 - x1
        card_depth = (focal_length_px * CREDIT_CARD_WIDTH) / card_width_px
        card_area_px = (x2 - x1) * (y2 - y1)
        per_pixel_area = CREDIT_CARD_AREA / card_area_px
        
        print(f"Card detected at [{x1}, {y1}, {x2}, {y2}], width: {card_width_px}px, depth: {card_depth:.2f}cm")
 
        # Food detection
        try:
            food_results = food_model(image, imgsz=640, conf=0.25, verbose=False)  # Lower confidence for food too
        except Exception as e:
            return jsonify({'error': f'Food detection failed: {str(e)}'}), 500
        
        if len(food_results[0].boxes.data) == 0:
            # Save debug image with card detection overlay
            debug_image = image.copy()
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_image, "Card Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            debug_path = os.path.join('tempupload', f'debug_no_food_{image_filename}')
            cv2.imwrite(debug_path, debug_image)
            
            return jsonify({
                'error': 'No food detected in the image', 
                'should_retry': True,
                'debug_image': os.path.basename(debug_path),
                'card_detected': True
            }), 400

        # Continue with depth estimation and food processing...
        # (Rest of the function remains the same)
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            input_batch = midas_transforms(pil_image).unsqueeze(0)
            if torch.cuda.is_available():
                input_batch = input_batch.half().cuda()
            
            with torch.no_grad():
                prediction = midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_image.shape[:2],
                    mode="nearest",
                ).squeeze()
            
            depth_map = prediction.float().cpu().numpy()
         
            if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
                
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-10)
            
            depth_vis_path = os.path.join('tempupload', f'depth_{image_filename}')
            plt.figure(figsize=(10, 8))
            plt.imshow(depth_norm, cmap=cm.jet)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(depth_vis_path)
            plt.close()
            
            card_depths = depth_map[y1:y2, x1:x2].flatten()
            card_depths = card_depths[~np.isnan(card_depths) & ~np.isinf(card_depths)]
            if len(card_depths) == 0:
                return jsonify({'error': 'Unable to determine card depth'}), 400
                
            card_depth_value = np.median(card_depths)  
            depth_scale = card_depth / card_depth_value
            real_depth_map = depth_map * depth_scale
        
            corners = [
                real_depth_map[0:50, 0:50],
                real_depth_map[0:50, -50:],
                real_depth_map[-50:, 0:50],
                real_depth_map[-50:, -50:]
            ]
            
            background_depths = []
            for corner in corners:
                valid_depths = corner[~np.isnan(corner) & ~np.isinf(corner)]
                if len(valid_depths) > 0:
                    background_depths.append(np.median(valid_depths))
            
            if not background_depths:
                return jsonify({'error': 'Unable to determine background depth'}), 400
                
            background_depth = np.median(background_depths)
            food_items = []
            for box in food_results[0].boxes.data:
                try:
                    item = process_food_item(box, image, real_depth_map, per_pixel_area, background_depth)
                    if item:
                        food_items.append(item)
                except Exception as e:
                    print(f"Error processing food item: {e}")
                    continue  
            
            if not food_items:
                return jsonify({'error': 'Could not process any food items'}), 400
           
            # Draw annotations
            for food in food_items:
                x1_f, y1_f, x2_f, y2_f = food['bbox']
                cv2.rectangle(image, (x1_f, y1_f), (x2_f, y2_f), (0, 255, 0), 2)
                label = f"{food['class']}: {food['volume_cm3']} cm³"
                if food['weight_g']:
                    label += f", {food['weight_g']}g"
                cv2.putText(image, label, (x1_f, y1_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Also draw credit card for reference
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, "Credit Card", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            annotated_image_path = os.path.join('tempupload', 'annotated_' + os.path.basename(image_path))
            cv2.imwrite(annotated_image_path, image)
            food_items = convert_numpy_types(food_items)
        
            # Calculate totals
            total_volume = float(round(sum(item.get('volume_cm3', 0) or 0 for item in food_items), 2))
            total_weight = float(round(sum(item.get('weight_g', 0) or 0 for item in food_items), 2))
            total_calories = float(round(sum(item.get('calories', 0) or 0 for item in food_items), 2))
            total_protein = float(round(sum(item.get('protein_g', 0) or 0 for item in food_items), 2))
            total_carbs = float(round(sum(item.get('carbs_g', 0) or 0 for item in food_items), 2))
            total_fats = float(round(sum(item.get('fats_g', 0) or 0 for item in food_items), 2))
            total_fiber = float(round(sum(item.get('fiber_g', 0) or 0 for item in food_items), 2))
            
            processing_time = time.time() - start_time
            
            return jsonify({
                'success': True,
                'food_items': food_items,
                'total_volume': total_volume,
                'total_weight': total_weight,
                'total_calories': total_calories,
                'total_protein': total_protein,
                'total_carbs': total_carbs,
                'total_fats': total_fats,
                'total_fiber': total_fiber,
                'annotated_image': os.path.basename(annotated_image_path),
                'depth_map': os.path.basename(depth_vis_path),
                'processing_time': processing_time,
                'card_confidence': float(card_box[4])
            })
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error in depth processing: {e}")
            print(traceback_str)
            return jsonify({'error': f'Depth processing failed: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in process_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    


@app.route('/get_food_history')
@login_required
def get_food_history():
    try:
        email = session['user']['email']
        date = request.args.get('date', datetime.utcnow().strftime('%Y-%m-%d'))
        
        entries = list(db.food_entries.find({
            'user_email': email,
            'date': date
        }).sort('timestamp', -1))
        
        for entry in entries:
            entry['_id'] = str(entry['_id'])
            entry['timestamp'] = entry['timestamp'].isoformat()
        
        return jsonify({'success': True, 'entries': entries})
    except Exception as e:
        print(f"Error in get_food_history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

def calculate_nutrition_targets(profile):
    if profile['sex'] == 'male':
        bmr = 88.362 + (13.397 * float(profile['weight'])) + (4.799 * float(profile['height'])) - (5.677 * float(profile['age']))
    else:
        bmr = 447.593 + (9.247 * float(profile['weight'])) + (3.098 * float(profile['height'])) - (4.330 * float(profile['age']))
    
    activity_level = determine_activity_level(profile)
   
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    
    tdee = bmr * activity_multipliers.get(activity_level, 1.2)
    
    if profile['goal'] == 'weight_loss':
        tdee *= 0.85
    elif profile['goal'] == 'weight_gain':
        tdee *= 1.15
    
    return {
        'activity_level': activity_level,  
        'target_calories': round(tdee),
        'target_protein': round((tdee * 0.3) / 4),
        'target_carbs': round((tdee * 0.4) / 4),
        'target_fats': round((tdee * 0.3) / 9)
    }

def determine_activity_level(profile):
    occupation_type = profile.get('occupation_type', '').lower()
    exercise_duration = float(profile.get('exercise_duration', 0))
    exercise_intensity = profile.get('exercise_intensity', 'low').lower()
    daily_steps = int(profile.get('daily_steps', 5000))
    weekly_exercise = int(profile.get('weekly_exercise', 0))
    
    explicit_activity_level = profile.get('activity_level', '').lower()
    valid_levels = ['sedentary', 'light', 'moderate', 'active', 'very_active']
    
    if explicit_activity_level in valid_levels:
        return explicit_activity_level

    if daily_steps >= 12500 or (exercise_duration >= 60 and exercise_intensity == 'high'):
        return 'very_active'
    elif daily_steps >= 10000 or (exercise_duration >= 30 and weekly_exercise >= 5):
        return 'active'
    elif daily_steps >= 7500 or (exercise_duration >= 20 and weekly_exercise >= 3):
        return 'moderate'
    elif daily_steps >= 5000 or weekly_exercise >= 1:
        return 'light'
    else:
        return 'sedentary'
    
def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    try:
        temp_dir = 'tempupload'
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
    except Exception as e:
        print(f"Error in cleanup: {e}")

# Call cleanup periodically (you might want to use a background task for this)
import atexit
atexit.register(cleanup_temp_files)

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert_path = 'cert.pem'
    key_path = 'key.pem'
    context.load_cert_chain(cert_path, key_path)
    
    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=True)