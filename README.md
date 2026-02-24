---
title: DreamTuner Studio
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
---

# DreamTuner Studio

An AI-powered image generation, analysis, and editing studio built with Gradio.

## Features

- 🎨 **Generate Images** - Create diverse images using Gemini 2.5 Flash
- 🔍 **Object Detection** - Detect and crop objects using OWL-ViT
- 📊 **Attribute Analysis** - Analyze images with custom opposite pairs using CLIP
- ✏️ **Image Editing** - Edit images with text instructions
- 📱 **Dual Interface** - Use via Gradio UI or React frontend
- 🎯 **Professional UI** - Compact, glass-morphism design

## Setup

1. **Get a Gemini API key** from [Google AI Studio](https://aistudio.google.com/)
2. **Add your API key** as a secret in Hugging Face Spaces:
   - Variable name: `GEMINI_API_KEY`
   - Your API key as the value
3. **The app will automatically load** all required models on first run

## Models Used

- **Google OWL-ViT** - For object detection and localization
- **OpenAI CLIP** - For image-text similarity and attribute analysis  
- **Google Gemini 2.5 Flash** - For image generation and editing

## Usage

### Through Gradio UI:
1. Click **"🚀 Init Engine"** to load AI models
2. **Enter a prompt** and generate images
3. **Use object detection** to find and crop specific objects
4. **Analyze images** with different attribute pairs (cute/ugly, happy/sad, etc.)
5. **Edit images** with text instructions

### Through React Frontend:
- Connect your React app using the provided API endpoints
- Available at: `https://your-username-dreamtuner-studio.hf.space`

## API Endpoints

The space provides REST API endpoints for integration:

- `POST /generate` - Generate images from text
- `POST /edit` - Edit images with text instructions  
- `POST /detect` - Detect objects in images
- `POST /analyze` - Analyze image attributes

## Configuration

This Space uses:
- **GPU accelerator** for fast inference
- **Gradio SDK 4.0.0** for the web interface
- **Python 3.8+** with PyTorch and Transformers
- **Automatic model caching** for faster loading

## Local Development

```bash
pip install -r requirements.txt
python app.py