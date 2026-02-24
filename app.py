# pip install -q gradio transformers accelerate torchvision pillow matplotlib xformers huggingface_hub requests

from click import style
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import gradio as gr
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import requests
import io
import base64
import time
from huggingface_hub import login

# -------- Memory Management --------
import gc

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# -------- device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Import after potential HF login
from transformers import OwlViTProcessor, OwlViTForObjectDetection, CLIPProcessor, CLIPModel

# -------- Customizable Opposite Word Pairs --------
CUSTOM_OPPOSITE_PAIRS = {
    "cute/ugly": ["a cute adorable lovely image", "an ugly scary disgusting image"],
    "happy/sad": ["a very happy and joyful image, extremely cheerful, full of delight and positive emotions, showing clear happiness",
    "a sad depressing gloomy crying mournful image"],
    "young/old": ["an old elderly aged image", "a young youthful childlike image"],
    "big/small": ["a big large massive image", "a small tiny little image"],
    "beautiful/ugly": ["a beautiful stunning gorgeous image", "an ugly disgusting hideous image"],
    "healthy/sick": ["a healthy vibrant thriving image", "a sick unhealthy decaying image"],
    "rich/poor": ["a rich luxurious expensive image", "a poor cheap low-quality image"],
    "fast/slow": ["a fast rapid speedy image", "a slow sluggish delayed image"],
    "modern/old": ["a modern contemporary futuristic image", "an old antique vintage image"],
    "clean/dirty": ["a clean pristine spotless image", "a dirty messy polluted image"],
    "bright/dark": ["a bright well-lit vibrant image", "a dark poorly-lit gloomy image"],
    "sharp/blurry": ["a sharp clear focused image", "a blurry fuzzy unfocused image"],
    "real/fake": ["a real authentic genuine image", "a fake artificial unreal image"],
    "natural/artificial": ["a natural organic authentic image", "an artificial synthetic fake image"],
    "peaceful/chaotic": ["a peaceful calm serene image", "a chaotic busy stressful image"],
    "warm/cold": ["a warm cozy inviting image", "a cold icy distant image"],
    "soft/hard": ["a soft gentle smooth image", "a hard rough sharp image"],
    "simple/complex": ["a simple minimal clean image", "a complex detailed busy image"],
    "safe/dangerous": ["""Secure_Place, Physical_Barriers, Supervised_Area, Safety_Rules, Limited_Access, Docile_Animal, Calm_Behavior, Professional_Help, 
                        Non_Aggressive, Relaxed_Body, Contained_Safe, Proper_Tools, Kitchen_Work, Toy_Weapon, Peaceful_Action, Gentle_Face, Domestic_Area""",
                        """Unsecured_Place, Free_Access, Unsupervised, No_Barriers, Dangerous_Area, Aggressive_Animal, Threat_Display, Violent_Action, Hostile_Posture, Unrestrained, 
                        Weapon_Threat, Knife_Attack, Gun_Violence, Violent_Fight, Teeth_Visible, Tense_Body, Wild_Nature"""],
    "friendly/aggressive": ["Play_Invitation, Affectionate_Contact, Social_Greeting, Positive_Vocalization, Relaxed_Body, Submissive_Posture, Non_Threatening_Gesture, Approach_Inviting, Juvenile_Behavior, Limited_Intensity, Gentle_Interaction, Trusting_Behavior, Cooperative_Action, Playful_Curiosity ", 
                            "Attack_Posture, Threat_Display, Defensive_Stance, Aggressive_Advance, Biting_Action, Charging_Movement, Fighting_Behavior, Predatory_Focus, Adult_Strength, Full_Intensity, Territorial_Defense, Harm_Intending, Fear_Response, Lethal_Capability "]
}

# -------- Utility Functions --------
def is_black_image(image: Image.Image, threshold=10):
    """Check if image is mostly black"""
    try:
        np_image = np.array(image)
        if len(np_image.shape) == 3:
            np_image = np_image.mean(axis=2)
        return np_image.mean() < threshold
    except:
        return False

def create_colored_fallback_image(message, index):
    """Create colored fallback images to verify display works"""
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    color = colors[index % len(colors)]

    img = Image.new('RGB', (512, 512), color=color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            pass
    except:
        font = ImageFont.load_default()

    draw.text((50, 200), "Image Display Test", fill='darkblue', font=font)
    draw.text((50, 240), message, fill='black', font=font)
    draw.text((50, 280), f"Color: {color}", fill='darkred', font=font)

    return img

# -------- Gemini API Integration --------
class GeminiAPI:
    def __init__(self):
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        self.api_key = None
        self.max_retries = 5

    def set_api_key(self, api_key):
        self.api_key = api_key
        return self.api_key is not None

    def pil_to_base64(self, img: Image.Image) -> str:
        """Converts a PIL Image object to a Base64 encoded string (PNG format)."""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def base64_to_pil(self, b64_string: str) -> Image.Image:
        """Converts a Base64 encoded string to a PIL Image object."""
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))

    def call_gemini_api(self, payload: dict) -> str:
        """Makes a POST request to the Gemini API with exponential backoff."""
        url = f"{self.api_url}?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                # Extract the base64 data from the response structure
                candidate = result.get('candidates', [{}])[0]
                part = candidate.get('content', {}).get('parts', [{}])[0]
                base64_data = part.get('inlineData', {}).get('data')

                if base64_data:
                    return base64_data
                else:
                    text_response = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No image generated.')
                    raise ValueError(f"API did not return image data. Response Text: {text_response}")

            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Attempt {attempt + 1} failed (Retrying in {wait_time}s): {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Final attempt failed. Error: {e}")
                    raise Exception(f"Image generation failed after multiple retries. Details: {e}")
        return ""

    def generate_image(self, prompt: str, aspect_ratio="1:1"):
        """Generate image using Gemini 2.5 Flash API - Text to Image"""
        if not self.api_key:
            raise ValueError("API key not set")

        # Map aspect ratio to Gemini format
        aspect_map = {
            "1:1": "1:1",
            "3:4": "3:4", 
            "4:3": "4:3",
            "9:16": "9:16",
            "16:9": "16:9"
        }
        
        gemini_aspect = aspect_map.get(aspect_ratio, "1:1")

        # Configuration for image output
        config = {
            "responseModalities": ["IMAGE"],
            "image_config": {
                "aspect_ratio": gemini_aspect
            }
        }

        # Prepare payload for text-to-image
        contents = [{
            "parts": [{"text": prompt}]
        }]

        payload = {
            "contents": contents,
            "generationConfig": config
        }
        
        try:
            base64_data = self.call_gemini_api(payload)
            image = self.base64_to_pil(base64_data)
            return [image]
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")

    def edit_image(self, prompt: str, image: Image.Image, aspect_ratio="1:1"):
        """Edit image using Gemini 2.5 Flash API - Image to Image"""
        if not self.api_key:
            raise ValueError("API key not set")

        # Map aspect ratio to Gemini format
        aspect_map = {
            "1:1": "1:1",
            "3:4": "3:4", 
            "4:3": "4:3",
            "9:16": "9:16",
            "16:9": "16:9"
        }
        
        gemini_aspect = aspect_map.get(aspect_ratio, "1:1")

        # Configuration for image output
        config = {
            "responseModalities": ["IMAGE"],
            "image_config": {
                "aspect_ratio": gemini_aspect
            }
        }

        # Convert image to base64
        base64_image = self.pil_to_base64(image)

        # Prepare payload for image-to-image (editing)
        contents = [{
            "parts": [
                {"text": prompt},
                {"inlineData": {
                    "mimeType": "image/png",
                    "data": base64_image
                }}
            ]
        }]

        payload = {
            "contents": contents,
            "generationConfig": config
        }
        
        try:
            base64_data = self.call_gemini_api(payload)
            edited_image = self.base64_to_pil(base64_data)
            return [edited_image]
            
        except Exception as e:
            raise Exception(f"Gemini editing API call failed: {str(e)}")

# -------- Model Loading --------
def load_models():
    global owl_processor, owl_model, clip_processor, clip_model, gemini_api

    gemini_api = GeminiAPI()
    
    # Use environment variable for API key (for Hugging Face deployment)
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBCBWt2SQD36Cvh0E-ytt-roVF1If6kNOc")
    if api_key:
        gemini_api.set_api_key(api_key)
        print("✅ Gemini API key set")
    else:
        print("⚠️  No Gemini API key found - image generation will not work")

    try:
        print("Loading OWL-ViT...")
        owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
        print("✅ OWL-ViT loaded")

        print("Loading CLIP...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        print("✅ CLIP loaded")

        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# -------- Enhanced CLIP Evaluation System --------
def proper_clip_similarity(image: Image.Image, text: str) -> float:
    """Calculate proper CLIP similarity between image and text"""
    try:
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logit_scale = clip_model.logit_scale.exp()
            similarity = F.cosine_similarity(outputs.image_embeds, outputs.text_embeds) * logit_scale
            return float(similarity.cpu().item())

    except Exception as e:
        print(f"CLIP similarity error: {e}")
        return 0.0

def research_clip_evaluation(image: Image.Image, positive_prompt: str, negative_prompt: str):
    """Comprehensive CLIP evaluation with proper scoring"""
    try:
        pos_similarity = proper_clip_similarity(image, positive_prompt)
        neg_similarity = proper_clip_similarity(image, negative_prompt)

        exp_pos = np.exp(pos_similarity)
        exp_neg = np.exp(neg_similarity)
        softmax_score = exp_pos / (exp_pos + exp_neg)

        max_score = max(pos_similarity, neg_similarity)
        min_score = min(pos_similarity, neg_similarity)
        normalized_diff = 0.5 if max_score == min_score else (pos_similarity - min_score) / (max_score - min_score)

        raw_diff = pos_similarity - neg_similarity

        return {
            'positive_similarity': pos_similarity,
            'negative_similarity': neg_similarity,
            'softmax_score': softmax_score,
            'normalized_difference': normalized_diff,
            'raw_difference': raw_diff,
            'confidence': abs(raw_diff)
        }

    except Exception as e:
        print(f"Research evaluation error: {e}")
        return {
            'positive_similarity': 0.0, 'negative_similarity': 0.0, 'softmax_score': 0.5,
            'normalized_difference': 0.5, 'raw_difference': 0.0, 'confidence': 0.0
        }

def create_custom_plot(results, positive_label, negative_label):
    """Create a customized plot for the selected opposite pair"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Add space for title above the plot
    plt.subplots_adjust(top=0.85)
    
    softmax = results['softmax_score']
    confidence = results['confidence']

    # Softmax distribution
    x = np.linspace(0, 1, 200)
    y = np.exp(-((x - softmax) * 8) ** 2)

    # Color based on score
    if softmax > 0.7:
        color = '#10b981'  # green
    elif softmax > 0.6:
        color = '#84cc16'  # lightgreen
    elif softmax > 0.4:
        color = '#f59e0b'  # orange
    else:
        color = '#ef4444'  # red

    ax.plot(x, y, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(x, y, alpha=0.3, color=color)
    ax.axvline(softmax, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Score: {softmax:.3f}')

    ax.set_xlabel(f'{negative_label} ⬅️ Neutral ➡️ {positive_label}', fontsize=10, fontweight='bold', color='#333333')
    ax.set_ylabel('Density', fontsize=9, color='#333333')
    
    # Move title above the plot
    ax.set_title(f'{positive_label} vs {negative_label}', 
                fontsize=12, fontweight='bold', pad=15, color='#333333')
    
    ax.legend(fontsize=9, facecolor='#f9fafb', edgecolor='#e5e7eb')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#333333')
    
    # Set light background
    ax.set_facecolor('#f9fafb')
    fig.patch.set_facecolor('#ffffff')

    return fig

def analyze_with_custom_pair(image, opposite_pair_key):
    """Analyze image with selected opposite pair"""
    if opposite_pair_key not in CUSTOM_OPPOSITE_PAIRS:
        opposite_pair_key = "cute/ugly"

    positive_prompt, negative_prompt = CUSTOM_OPPOSITE_PAIRS[opposite_pair_key]
    positive_label, negative_label = opposite_pair_key.split('/')

    results = research_clip_evaluation(image, positive_prompt, negative_prompt)
    fig = create_custom_plot(results, positive_label, negative_label)

    return results, fig, positive_label, negative_label

def generate_custom_report(results, positive_label, negative_label, detection_score=None, compact=False):
    """Generate detailed report for custom analysis"""
    score = results['softmax_score']
    confidence = results['confidence']

    if score > 0.7:
        rating = "⭐⭐⭐⭐⭐ Strongly Positive"
    elif score > 0.6:
        rating = "⭐⭐⭐⭐ Positive"
    elif score > 0.5:
        rating = "⭐⭐⭐ Neutral"
    elif score > 0.4:
        rating = "⭐⭐ Negative"
    else:
        rating = "⭐ Strongly Negative"

    # For compact mode (frontend) - ONLY SHOW INTERPRETATION
    if compact:
        if score > 0.7:
            report = f"💡 **Interpretation:** The image predominantly exhibits **{positive_label}** characteristics"
        elif score > 0.6:
            report = f"💡 **Interpretation:** The image shows **{positive_label}** qualities"
        elif score > 0.5:
            report = f"💡 **Interpretation:** The image is balanced between {positive_label} and {negative_label}"
        elif score > 0.4:
            report = f"💡 **Interpretation:** The image shows **{negative_label}** qualities"
        else:
            report = f"💡 **Interpretation:** The image predominantly exhibits **{negative_label}** characteristics"
        
        return report
    else:
        report = f"""
📊 **Analysis Results:**
• **{positive_label} Score:** {results['positive_similarity']:.3f}
• **{negative_label} Score:** {results['negative_similarity']:.3f}
• **Overall Score:** {score:.3f}
• **Confidence:** {confidence:.3f}

🏆 **Rating:** {rating}"""

        return report

# -------- Image Processing Functions --------
def enhance_image_quality(image: Image.Image):
    """Simple image enhancement"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        contrast_factor = 1.1
        brightness_factor = 1.1

        np_image = np.array(image)
        image_mean = np_image.mean()

        if image_mean < 80:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

        return image
    except Exception as e:
        print(f"Image enhancement error: {e}")
        return image

def is_dark_image(image: Image.Image, threshold=60):
    try:
        np_image = np.array(image)
        if len(np_image.shape) == 3:
            np_image = np_image.mean(axis=2)
        return np_image.mean() < threshold
    except:
        return False

def is_low_contrast(image: Image.Image, threshold=25):
    try:
        np_image = np.array(image)
        if len(np_image.shape) == 3:
            np_image = np_image.mean(axis=2)
        return np_image.std() < threshold
    except:
        return False

# -------- Image Generation --------
def generate_diverse_images(prompt, num_images, aspect_ratio):
    prompt_variations = [
        prompt, f"{prompt}, detailed background", f"{prompt}, different angle",
        f"{prompt}, unique perspective", f"{prompt}, creative interpretation"
    ]

    print(f"Generating {num_images} images using Gemini 2.5 Flash for: {prompt}")

    images = []
    status = "🖼 🎨🖍️🧑‍🎨 Generation completed successfully"
    enhanced_count = 0

    for i in range(int(num_images)):
        try:
            prompt_variation = prompt_variations[i % len(prompt_variations)]
            generated_images = gemini_api.generate_image(
                prompt=prompt_variation, aspect_ratio=aspect_ratio
            )

            if generated_images and len(generated_images) > 0:
                img = generated_images[0]
                needs_enhancement = is_dark_image(img) or is_low_contrast(img)
                if needs_enhancement:
                    enhanced_img = enhance_image_quality(img)
                    enhanced_count += 1
                else:
                    enhanced_img = img
                images.append(enhanced_img)
            else:
                fallback = create_colored_fallback_image(f"No image #{i+1}", i)
                images.append(fallback)

        except Exception as e:
            print(f"Gemini generation error for image {i+1}: {e}")
            fallback = create_colored_fallback_image(f"Error: {str(e)[:50]}", i)
            images.append(fallback)

    if enhanced_count > 0:
        status = f"🐦‍🔥 Generated {len(images)} images ({enhanced_count} enhanced)"
    else:
        status = f"🐦‍🔥 Generated {len(images)} high-quality images"

    return images, prompt, status

# -------- Image Editing Function --------
def edit_selected_image(edit_prompt, current_img, aspect_ratio, original_img):
    """Edit a selected image using Gemini API"""
    if current_img is None:
        raise ValueError("No image selected for editing")
    
    print(f"Editing image with prompt: {edit_prompt}")
    
    try:
        edited_images = gemini_api.edit_image(
            prompt=edit_prompt, 
            image=current_img, 
            aspect_ratio=aspect_ratio
        )

        if edited_images and len(edited_images) > 0:
            edited_img = edited_images[0]
            return edited_img, current_img, original_img, edited_img, "✅ Image edited successfully!"
        else:
            raise Exception("No image returned from editing API")
            
    except Exception as e:
        error_msg = f"❌ Editing failed: {str(e)}"
        print(f"Editing error: {e}")
        return current_img, current_img, original_img, current_img, error_msg

# -------- Object Detection with NMS --------
def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping boxes"""
    if not detections:
        return []
    
    detections.sort(key=lambda x: x["score"], reverse=True)
    
    suppressed = []
    
    while detections:
        best = detections.pop(0)
        suppressed.append(best)
        
        detections = [
            det for det in detections 
            if calculate_iou(best["box"], det["box"]) < iou_threshold
            or best["label"] != det["label"]
        ]
    
    return suppressed

def owlvitz_detect(image: Image.Image, queries: list, threshold=0.12, padding_ratio=0.01):
    if not queries: 
        return []

    try:
        inputs = owl_processor(text=queries, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        processed = owl_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        boxes, scores, labels = processed.get("boxes", []), processed.get("scores", []), processed.get("labels", [])
        detections = []
        img_w, img_h = image.size
        
        pad_px = max(2, int(padding_ratio * min(img_w, img_h)))

        for i in range(len(scores)):
            try:
                score = float(scores[i].cpu().item())
                label_idx = int(labels[i].cpu().item())
                label_text = queries[label_idx] if label_idx < len(queries) else queries[0]
                box = boxes[i].cpu().numpy().tolist()
                xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
                
                xmin = max(0, xmin - pad_px)
                ymin = max(0, ymin - pad_px)
                xmax = min(img_w, xmax + pad_px)
                ymax = min(img_h, ymax + pad_px)
                
                if (xmax - xmin) > 10 and (ymax - ymin) > 10:
                    crop = image.crop((xmin, ymin, xmax, ymax))
                    detections.append({
                        "label": label_text, 
                        "score": score, 
                        "box": (xmin, ymin, xmax, ymax), 
                        "crop": crop
                    })
            except Exception as e:
                print(f"Detection processing error: {e}")
                continue

        detections = non_max_suppression(detections, iou_threshold=0.4)
        return detections

    except Exception as e:
        print(f"Detection error: {e}")
        return []

def draw_boxes(image: Image.Image, detections: list):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except:
            pass
    except:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline="#10b981", width=2)
        
        # Draw label background
        label_text = f"{det['label']} ({det['score']:.2f})"
        bbox = draw.textbbox((x0, max(0, y0 - 18)), label_text, font=font)
        draw.rectangle(bbox, fill="#10b981")
        
        # Draw label text
        draw.text((x0, max(0, y0 - 18)), label_text, fill="white", font=font)

    return out


# -------- ULTRA COMPACT CSS - Professional single-page layout --------
compact_css = """
/* Remove ALL padding and margins from the entire interface */
.gradio-container {
    padding: 0px !important;
    margin: 0px !important;
    max-width: 100% !important;
}

/* Remove padding from main container */
.container {
    padding: 0px !important;
    margin: 0px !important;
    max-width: 100% !important;
}

/* Remove padding from all panels */
.left-panel,
.middle-panel, 
.right-panel {
    padding: 0px !important;
    margin: 0px !important;
    gap: 1px !important;
}

/* Remove padding from groups */
.gr-group {
    padding: 2px !important;
    margin: 0px !important;
    border: none !important;
    background: transparent !important;
}

/* Remove form gaps */
.form {
    gap: 1px !important;
    padding: 0px !important;
}

/* Ultra compact and cool buttons */
.gr-button {
    padding: 2px 4px !important;
    font-size: 10px !important;
    margin: 1px !important;
    min-height: auto !important;
    border-radius: 4px !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.2s ease !important;
}

.gr-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.gr-button:active {
    transform: translateY(0px) !important;
}

/* Primary button styling */
.gr-button.primary {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
    color: white !important;
    border: none !important;
}

.gr-button.secondary {
    background: linear-gradient(135deg, #f1f5f9, #e2e8f0) !important;
    color: #475569 !important;
}

/* Larger prompt text area */
.gr-textbox textarea {
    min-height: 60px !important;  /* Increased height */
    padding: 5px 8px !important;
    font-size: 12px !important;
    border-radius: 4px !important;
    line-height: 1.4 !important;
}

/* Compact dropdowns */
.gr-dropdown {
    min-height: auto !important;
    font-size: 11px !important;
}

/* Remove spacing from rows */
.gr-row {
    gap: 1px !important;
    padding: 0px !important;
    margin: 0px !important;
}

/* Remove spacing from columns */
.gr-column {
    gap: 1px !important;
    padding: 0px !important;
}

/* Compact gallery */
.gr-gallery {
    gap: 1px !important;
}

/* Remove borders and shadows */
.panel {
    border: none !important;
    box-shadow: none !important;
}

/* Ensure images fit compactly */
.gr-image {
    border: none !important;
    padding: 0px !important;
}

/* Remove any remaining gaps in markdown */
.gr-markdown {
    padding: 0px !important;
    margin: 0px !important;
    font-size: 12px !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
    margin: 2px 0 !important;
    padding: 0 !important;
}

/* Compact plot */
.gr-plot {
    padding: 0px !important;
    margin: 0px !important;
}

/* Remove any background colors that create visual separation */
.gr-group,
.panel {
    background: transparent !important;
}

/* Ensure the entire app uses full width */
#root {
    padding: 0px !important;
    margin: 0px !important;
}

/* Compact sliders */
.gr-slider {
    padding: 0px !important;
    margin: 0px !important;
}

.gr-slider .gr-text {
    font-size: 10px !important;
}

/* Status text styling */
.gr-textbox {
    font-size: 10px !important;
    min-height: 20px !important;
}

/* Make labels compact */
.gr-form > .gr-form > .wrap > .block > .wrap > .text-gray-500 {
    font-size: 10px !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Compact group headers */
.gr-group > .gr-markdown {
    font-size: 11px !important;
    font-weight: 600 !important;
    margin-bottom: 3px !important;
}

/* Left panel specific - remove extra space */
.left-panel .gr-group {
    padding: 1px !important;
    margin: 0px !important;
}

.left-panel .gr-form {
    gap: 1px !important;
}

/* Middle panel - larger images */
.middle-panel .gr-image {
    height: 350px !important;  /* Increased height */
    min-height: 350px !important;
}

/* Reduce spacing in left panel sections */
.left-panel .gr-group .gr-markdown {
    margin-bottom: 2px !important;
}
"""

# Initialize models as None
owl_processor = None
owl_model = None
clip_processor = None
clip_model = None
gemini_api = None

# Pre-load models at startup for Hugging Face
print("🚀 Starting DreamTuner Studio...")
print("Loading AI models... This may take a moment.")

# -------- Ultra Compact Gradio UI --------
with gr.Blocks(
    title="DreamTuner: Compact Studio", 
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="emerald"),
    css=compact_css
) as demo:
    
    gr.Markdown("""
    # 🎨 DreamTuner Studio - Generate, analyze, and refine images with AI-powered tools
    """)
    
    # State variables
    gallery_state = gr.State([])
    current_detections_state = gr.State([])
    previous_detections_state = gr.State([])
    selected_idx_state = gr.State(None)
    selected_crop_idx_state = gr.State(None)
    models_loaded = gr.State(False)

    with gr.Row(equal_height=True):
        # ========== LEFT PANEL: Configuration ==========
        with gr.Column(scale=1, min_width=280, elem_classes="left-panel"):
            # Engine Configuration
            with gr.Group():
                gr.Markdown("#### ⚙️ Engine")
                
                with gr.Row():
                    load_models_btn = gr.Button("🚀 Init Engine", variant="primary", size="sm")
                    load_status = gr.Textbox(
                        value="🟡 Ready to load", 
                        interactive=False,
                        show_label=False
                    )

            # Image Generation
            with gr.Group():
                gr.Markdown("#### 🎨 Generation")
                
                prompt_input = gr.Textbox(
                    label="💭 Prompt",
                    value="a cute sad baby is playing with a cute fluffy pomeranian puppy but an ugly monster is hiding behind a tree in a park",
                    lines=3,  # Increased lines for larger prompt box
                    placeholder="Describe the image...",
                    max_lines=4
                )

                with gr.Row():
                    aspect_ratio_dropdown = gr.Dropdown(
                        choices=["1:1", "3:4", "4:3", "9:16", "16:9"],
                        value="1:1",
                        label="📐 Aspect"
                    )
                    num_images_slider = gr.Slider(
                        1, 8, value=4, step=1, 
                        label="🖼️ Count"
                    )

                generate_btn = gr.Button("✨ Generate", variant="primary", size="sm")
                gen_status = gr.Textbox(
                    value="Ready", 
                    interactive=False,
                    show_label=False
                )

            # Editing
            with gr.Group():
                gr.Markdown("#### ✏️ Editing")
                
                edit_prompt_input = gr.Textbox(
                    label="✏️ Edit Instructions",
                    placeholder="Describe changes to make to the current image...",
                    lines=1
                )
                
                with gr.Row():
                    edit_btn = gr.Button("🔄 Apply", variant="primary", size="sm")
                    reset_editing_btn = gr.Button("↩️ Reset", variant="secondary", size="sm")
                
                edit_status = gr.Textbox(
                    value="Select image first", 
                    interactive=False,
                    show_label=False
                )

        # ========== MIDDLE PANEL: Image Workspace ==========
        with gr.Column(scale=3, min_width=600, elem_classes="middle-panel"):  # Increased scale for middle panel
            # Image Grid
            with gr.Group():
                gr.Markdown("#### 🖼️ Image Workspace")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🎨 Generated**")
                        generated_image = gr.Image(
                            type="pil",
                            interactive=False,
                            show_download_button=True,
                            height=350  # Increased height
                        )
                    
                    with gr.Column():
                        gr.Markdown("**🔍 Detected**")
                        detected_image = gr.Image(
                            type="pil",
                            interactive=False,
                            show_download_button=True,
                            height=350  # Increased height
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**🔄 Current**")
                        current_image = gr.Image(
                            type="pil",
                            interactive=False,
                            show_download_button=True,
                            height=350  # Increased height
                        )
                    
                    with gr.Column():
                        gr.Markdown("**📜 Previous**")
                        previous_image = gr.Image(
                            type="pil",
                            interactive=False,
                            show_download_button=True,
                            height=350  # Increased height
                        )

        # ========== RIGHT PANEL: Analysis & Tools ==========
        with gr.Column(scale=1, min_width=280, elem_classes="right-panel"):
            # Object Detection
            with gr.Group():
                gr.Markdown("#### 🕵️ Detection")
                
                object_input = gr.Textbox(
                    label="🔎 Objects to Detect",
                    value="person, animal, tree, monster, puppy, baby",
                    placeholder="Comma-separated objects...",
                    lines=1
                )

                with gr.Row():
                    threshold_slider = gr.Slider(
                        0.05, 0.5, value=0.15, step=0.01, 
                        label="🎯 Threshold"
                    )
                
                with gr.Row():
                    detect_current_btn = gr.Button("🔍 Current", variant="primary", size="sm")
                    detect_previous_btn = gr.Button("🔍 Previous", variant="secondary", size="sm")
                
                detect_status = gr.Textbox(
                    value="Ready", 
                    interactive=False,
                    show_label=False
                )

                crops_gallery = gr.Gallery(
                    label="📦 Detected Objects",
                    columns=3,
                    height=120,
                    object_fit="cover",
                    show_download_button=False
                )

            # Analysis
            with gr.Group():
                gr.Markdown("#### 📊 Analysis")
                
                opposite_dropdown = gr.Dropdown(
                    choices=list(CUSTOM_OPPOSITE_PAIRS.keys()),
                    value="cute/ugly",
                    label="⚖️ Attributes"
                )
                
                with gr.Row():
                    analyze_current_btn = gr.Button("🌆 Current", variant="secondary", size="sm")
                    analyze_previous_btn = gr.Button("📜 Previous", variant="secondary", size="sm")
                    analyze_crop_btn = gr.Button("🌠 Object", variant="secondary", size="sm")

                eval_plot = gr.Plot(
                    show_label=False
                )
                
                eval_report = gr.Textbox(
                    lines=2, 
                    interactive=False,
                    show_label=False,
                    label="💡 Interpretation"
                )

    # --- Event Handlers ---
    def update_pair_info(opposite_key):
        """Update pair information display"""
        if opposite_key in CUSTOM_OPPOSITE_PAIRS:
            positive_label, negative_label = opposite_key.split('/')
            return f"{opposite_key}: 🟢={positive_label}, 🔴={negative_label}"
        return "Invalid pair selected"

    def on_load_models():
        try:
            success = load_models()
            if success:
                return "🟢 Engine loaded", True
            else:
                return "🔴 Load failed", False
        except Exception as e:
            return f"🔴 Error: {str(e)}", False

    def on_models_loaded(loaded):
        interactive_state = gr.update(interactive=loaded)
        return (
            interactive_state,  # generate_btn
            interactive_state,  # detect_current_btn
            interactive_state,  # detect_previous_btn
            interactive_state,  # analyze_current_btn
            interactive_state,  # analyze_previous_btn
            interactive_state,  # analyze_crop_btn
            interactive_state   # edit_btn
        )

    def on_generate(prompt, num_images, aspect_ratio, models_loaded_flag):
        if not models_loaded_flag:
            return [], [], None, None, None, None, "🔴 Initialize engine first"

        try:
            images, enhanced_prompt, status = generate_diverse_images(
                prompt, int(num_images), aspect_ratio
            )
            
            if images and len(images) > 0:
                # Get the first generated image
                first_image = images[0]
                return images, images, first_image, first_image, first_image, first_image, status
            else:
                raise Exception("No images generated")
                
        except Exception as e:
            error_msg = f"🔴 Generation failed: {str(e)}"
            print(f"Generation error: {e}")
            test_image = create_colored_fallback_image("Test", 0)
            return [test_image], [test_image], test_image, test_image, test_image, test_image, error_msg

    def on_edit_image(edit_prompt, current_img, aspect_ratio, original_img):
        """Edit image"""
        if not edit_prompt.strip():
            return current_img, current_img, current_img, current_img, "🔴 Enter instructions"
        if current_img is None:
            return None, None, None, None, "🔴 Select image first"
        
        print(f"Editing image with: {edit_prompt}")
        
        try:
            edited_images = gemini_api.edit_image(
                prompt=edit_prompt, 
                image=current_img, 
                aspect_ratio=aspect_ratio
            )

            if edited_images and len(edited_images) > 0:
                edited_img = edited_images[0]
                print("🟢 Image editing successful")
                
                return (
                    edited_img,      # current_image
                    current_img,     # previous_image
                    edited_img,      # generated_image
                    edited_img,      # detected_image
                    "🟢 Edited successfully!"
                )
            else:
                raise Exception("No image returned from editing API")
            
        except Exception as e:
            error_msg = f"🔴 Editing failed: {str(e)}"
            print(f"Editing error: {e}")
            return current_img, current_img, current_img, current_img, error_msg

    def on_reset_editing(original_img):
        """Reset editing to original image"""
        if original_img is not None:
            print("Resetting editing to original image")
            return (
                original_img,  # current_image
                original_img,  # previous_image
                original_img,  # generated_image
                original_img,  # detected_image
                "🔄 Reset to original"
            )
        else:
            return None, None, None, None, "🔴 No original image"

    def on_detect_current(current_img, object_text, threshold, models_loaded_flag):
        """Object detection on current image"""
        return detect_objects(current_img, object_text, threshold, models_loaded_flag, "current")

    def on_detect_previous(previous_img, object_text, threshold, models_loaded_flag):
        """Object detection on previous image"""
        return detect_objects(previous_img, object_text, threshold, models_loaded_flag, "previous")

    def detect_objects(image, object_text, threshold, models_loaded_flag, image_type):
        """Generic object detection function"""
        if not models_loaded_flag:
            return [], "🔴 Models not loaded", [], image
        if image is None:
            return [], f"🔴 Select {image_type} image first", [], image
        if not object_text.strip():
            return [], "🔴 Enter objects to detect", [], image

        try:
            queries = [q.strip() for q in object_text.split(",") if q.strip()]

            expanded_queries = []
            for q in queries:
                if q.lower() in ["person", "man", "woman"]:
                    expanded_queries.extend(["person", "man", "woman", "human"])
                elif q.lower() in ["animal", "dog", "cat"]:
                    expanded_queries.extend(["animal", "dog", "cat", "pet"])
                else:
                    expanded_queries.append(q)

            queries = list(set(expanded_queries))
            detections = owlvitz_detect(image, queries, threshold=float(threshold))
            annotated = draw_boxes(image, detections)

            if not detections:
                status = f"🟡 No objects in {image_type}"
                crop_images = []
            else:
                status = f"🟢 Found {len(detections)} in {image_type}"
                crop_images = [det["crop"] for det in detections]

            detected_img = annotated if annotated else image

            # Store detections based on image type
            if image_type == "current":
                detections_state = detections
            else:
                detections_state = detections

            return crop_images, status, detections_state, detected_img

        except Exception as e:
            error_msg = f"🔴 Detection error: {str(e)}"
            print(f"Detection error details: {e}")
            return [], error_msg, [], image

    def on_analyze_current(current_image, opposite_key, models_loaded_flag):
        """Analyze current image"""
        if not models_loaded_flag:
            return None, "🔴 Models not loaded"
        if current_image is None:
            return None, "🔴 Select current image"

        try:
            results, fig, positive_label, negative_label = analyze_with_custom_pair(current_image, opposite_key)
            report = generate_custom_report(results, positive_label, negative_label, compact=True)

            return fig, report

        except Exception as e:
            error_msg = f"🔴 Analysis failed: {str(e)}"
            return None, error_msg

    def on_analyze_previous(previous_image, opposite_key, models_loaded_flag):
        """Analyze previous image"""
        if not models_loaded_flag:
            return None, "🔴 Models not loaded"
        if previous_image is None:
            return None, "🔴 No previous image"

        try:
            results, fig, positive_label, negative_label = analyze_with_custom_pair(previous_image, opposite_key)
            report = generate_custom_report(results, positive_label, negative_label, compact=True)

            return fig, report

        except Exception as e:
            error_msg = f"🔴 Analysis failed: {str(e)}"
            return None, error_msg

    def on_analyze_crop(detections_state, selected_crop_idx_state, opposite_key, models_loaded_flag):
        """Analyze specific crop"""
        if not models_loaded_flag:
            return None, "🔴 Models not loaded"
        if not detections_state or selected_crop_idx_state is None:
            return None, "🔴 Select object first"

        try:
            if selected_crop_idx_state < len(detections_state):
                det = detections_state[selected_crop_idx_state]
                crop_img = det["crop"]
                object_label = det["label"]
                detection_score = det["score"]

                results, fig, positive_label, negative_label = analyze_with_custom_pair(crop_img, opposite_key)
                report = generate_custom_report(results, positive_label, negative_label, detection_score, compact=True)

                return fig, report
            else:
                return None, "🔴 Invalid selection"

        except Exception as e:
            error_msg = f"🔴 Crop analysis failed: {str(e)}"
            print(f"Analyze crop error: {e}")
            return None, error_msg

    def on_crop_select(evt: gr.SelectData, detections_state):
        """Handle crop selection"""
        try:
            idx = int(evt.index)
            if detections_state and idx < len(detections_state):
                det = detections_state[idx]
                status_msg = f"Selected: {det['label']} ({det['score']:.3f})"
                return idx, status_msg
            else:
                return None, "No object selected"
        except Exception as e:
            print(f"Crop selection error: {e}")
            return None, f"Selection error: {str(e)}"

    # --- Connect Events ---
    load_models_btn.click(
        on_load_models,
        inputs=[],
        outputs=[load_status, models_loaded]
    ).then(
        on_models_loaded,
        inputs=[models_loaded],
        outputs=[generate_btn, detect_current_btn, detect_previous_btn, analyze_current_btn, analyze_previous_btn, analyze_crop_btn, edit_btn]
    )

    generate_btn.click(
        on_generate,
        inputs=[prompt_input, num_images_slider, aspect_ratio_dropdown, models_loaded],
        outputs=[gallery_state, gallery_state, generated_image, current_image, detected_image, previous_image, gen_status]
    )

    detect_current_btn.click(
        on_detect_current,
        inputs=[current_image, object_input, threshold_slider, models_loaded],
        outputs=[crops_gallery, detect_status, current_detections_state, detected_image]
    ).then(
        lambda: 0,
        outputs=[selected_crop_idx_state]
    )

    detect_previous_btn.click(
        on_detect_previous,
        inputs=[previous_image, object_input, threshold_slider, models_loaded],
        outputs=[crops_gallery, detect_status, previous_detections_state, detected_image]
    ).then(
        lambda: 0,
        outputs=[selected_crop_idx_state]
    )

    edit_btn.click(
        on_edit_image,
        inputs=[edit_prompt_input, current_image, aspect_ratio_dropdown, current_image],
        outputs=[current_image, previous_image, generated_image, detected_image, edit_status]
    )

    reset_editing_btn.click(
        on_reset_editing,
        inputs=[current_image],
        outputs=[current_image, previous_image, generated_image, detected_image, edit_status]
    )

    opposite_dropdown.change(
        update_pair_info,
        inputs=[opposite_dropdown],
        outputs=[]
    )

    analyze_current_btn.click(
        on_analyze_current,
        inputs=[current_image, opposite_dropdown, models_loaded],
        outputs=[eval_plot, eval_report]
    )

    analyze_previous_btn.click(
        on_analyze_previous,
        inputs=[previous_image, opposite_dropdown, models_loaded],
        outputs=[eval_plot, eval_report]
    )

    analyze_crop_btn.click(
        on_analyze_crop,
        inputs=[current_detections_state, selected_crop_idx_state, opposite_dropdown, models_loaded],
        outputs=[eval_plot, eval_report]
    )

    crops_gallery.select(
        on_crop_select,
        inputs=[current_detections_state],
        outputs=[selected_crop_idx_state, detect_status]
    )

# Launch 
if __name__ == "__main__":
    demo.launch(
        share=True,  
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
        quiet=True
    )