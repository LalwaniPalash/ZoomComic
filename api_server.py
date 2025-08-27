#!/usr/bin/env python3
"""
Flask API server for real-time fractal comic generation.
Provides OCR-based meta-narrative generation for infinite zoom comics.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from enhanced_model_utils import EnhancedOptimizedTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Global model instance
model = None

def initialize_model():
    """Initialize the enhanced quantized model on startup."""
    global model
    try:
        logger.info("Loading enhanced quantized model with micro-fiction techniques...")
        model = EnhancedOptimizedTransformer()
        model.load_model()
        logger.info("Enhanced model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load enhanced model: {e}")
        raise

@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the main HTML frontend."""
    return send_from_directory('.', 'viewer.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/generate', methods=['POST'])
def generate_continuity_narrative():
    """Generate narrative using continuity-based approach to prevent meta-commentary."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        zoom_level = data.get('zoom_level', 1)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        logger.info(f"Generating continuity text for level: {zoom_level}")
        
        # Generate using continuity-based approach
        generated_text = model.generate_continuity_text(
            level=zoom_level,
            max_new_tokens=30,
            temperature=0.9
        )
        
        # Get the complete story so far
        full_story = model.get_continuity_story()
        
        response = {
            'generated_text': generated_text,
            'full_text': generated_text,  # Just the new sentence
            'complete_story': full_story,  # The entire narrative
            'zoom_level': zoom_level,
            'meta_context': False  # No meta-commentary
        }
        
        logger.info(f"Generated: {generated_text}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-initial', methods=['POST'])
def generate_initial_comic():
    """Generate initial comic text and reset continuity."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Accept both 'prompt' and 'seed' for compatibility
        prompt = data.get('prompt', data.get('seed', '')).strip()
        
        if not prompt:
            return jsonify({'error': 'Prompt or seed is required'}), 400
            
        logger.info(f"Generating initial comic for prompt: {prompt}")
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Reset continuity and start fresh
        model.reset_continuity(prompt)
        
        # Generate first story beat
        generated_text = model.generate_continuity_text(
            level=1,
            max_new_tokens=30,
            temperature=0.9
        )
        
        response = {
            'generated_text': generated_text,
            'full_text': generated_text,  # Frontend expects this field
            'prompt': prompt,
            'zoom_level': 1,
            'complete_story': generated_text
        }
        
        logger.info(f"Generated initial: {generated_text}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Initial generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset_story', methods=['POST'])
def reset_story():
    """Reset the continuity system."""
    try:
        data = request.get_json()
        seed = data.get('seed', 'A sentient nebula dreams of a city made of light.') if data else 'A sentient nebula dreams of a city made of light.'
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        model.reset_continuity(seed)
        
        response = {
            'message': 'Story reset successfully',
            'seed': seed
        }
        
        logger.info(f"Story reset with seed: {seed}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Start the server
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)