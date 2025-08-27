# Zoom Comic - Infinite Fractal Comic Generator

A revolutionary comic creation tool that generates endless, zoomable comic strips using AI language models and interactive web technology. Start with a simple sentence and dive into an infinite universe of interconnected stories through a beautiful web interface.

## Demo

![Zoom Comic Demo](demo.gif)

## Features

- **Infinite Comic Generation**: Creates recursive comic panels that can be zoomed into infinitely
- **AI Micro-Fiction Engine**: Uses GPT-2 with advanced micro-fiction techniques including Borges-like style primer and semantic beam-pruning
- **Interactive Web Interface**: Smooth panning and zooming with real-time panel generation
- **Web-Based**: Modern HTML/CSS rendering with Flask API backend
- **Real-time Generation**: New panels appear instantly as you zoom deeper
- **Narrative Continuity**: Advanced continuity tracking prevents meta-commentary and maintains story coherence

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/LalwaniPalash/ZoomComic.git
cd ZoomComic
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Start Your First Comic

1. Start the API server:
```bash
source venv/bin/activate
python api_server.py
```

2. Open your browser and navigate to:
```
http://localhost:5001
```

3. Enter a seed sentence and start exploring! Click "Start New Exploration" and then zoom into panels to generate deeper levels.

## Usage

### Web Interface

The web interface provides an intuitive way to explore infinite comics:

- **Mouse**: Drag to pan, scroll to zoom
- **Panel Interaction**: Click on any panel to generate child panels and zoom deeper
- **Controls**: Use the interface buttons for:
  - Starting new explorations with custom seed text
  - Zooming to fit panels
  - Viewing generation logs
- **Real-time Generation**: New story content appears instantly as you explore

### API Endpoints

The Flask server provides REST API endpoints:

- `POST /generate-initial`: Start a new comic with a seed sentence
- `POST /generate`: Generate continuation text for deeper levels
- `GET /`: Serve the interactive web viewer

## How It Works

### The Infinite Generation Algorithm

1. **Seed Panel**: Starts with your input sentence in an interactive panel
2. **Enhanced Text Generation**: AI model with micro-fiction techniques generates continuation text
3. **Narrative Continuity**: Advanced continuity tracking maintains story coherence across levels
4. **Recursive Exploration**: Each panel can spawn child panels as you zoom deeper
5. **Infinite Depth**: Process repeats indefinitely with semantic beam-pruning for quality

### Technical Architecture

- **Language Model**: GPT-2 with enhanced micro-fiction techniques:
  - Borges-like style primer for surreal, literary tone
  - Semantic beam-pruning using sentence transformers
  - Recursive style grafting with repetition detection
  - Narrative continuity tracking to prevent meta-commentary
- **Backend**: Flask API server with CORS support
- **Frontend**: Modern HTML/CSS with interactive panels
- **Rendering**: Real-time DOM manipulation for smooth exploration
- **Optimization**: Model quantization and MPS/CUDA acceleration support

## Examples

Try these seed sentences for interesting results:

- `"A lonely pixel finds color."`
- `"The last library on Earth."`
- `"Dreams within dreams within dreams."`
- `"A robot learns to paint."`
- `"The universe is a story telling itself."`

## Advanced Usage

### Custom Model Configuration

You can modify the language model settings in `enhanced_model_utils.py`:

```python
# Adjust generation parameters in EnhancedOptimizedTransformer
self.generation_config = {
    "temperature": 0.9,     # Creativity level (0.1-1.5)
    "top_p": 0.9,          # Nucleus sampling
    "repetition_penalty": 1.3,  # Prevent repetition
    "num_return_sequences": 8,  # Candidates for beam-pruning
}
```

### Visual Customization

Modify the CSS styles in `viewer.html` to customize panel appearance:

```css
.panel {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 2px solid #333;
    font-family: 'Georgia', serif;
    border-radius: 8px;
}
```

### Performance Optimization

For better performance:

1. **GPU/MPS Acceleration**: Automatically uses CUDA or Apple Silicon MPS when available
2. **Model Caching**: Model loads once and stays in memory
3. **Lazy Generation**: Panels only generate when clicked
4. **Semantic Pruning**: Only the best candidates are selected using sentence transformers

## File Structure

```
ZoomComic/
├── api_server.py            # Flask API server
├── enhanced_model_utils.py  # Enhanced GPT-2 with micro-fiction techniques
├── viewer.html             # Interactive web viewer
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── venv/                  # Virtual environment (after setup)
└── model_cache/           # Cached models (after first run)
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space (~500MB)
   - Try running again (downloads resume automatically)

2. **API Server Won't Start**
   - Ensure virtual environment is activated: `source venv/bin/activate`
   - Check if port 5001 is available: `lsof -i :5001`
   - Install dependencies: `pip install -r requirements.txt`

3. **Web Interface Not Loading**
   - Verify API server is running on http://localhost:5001
   - Check browser console for JavaScript errors
   - Ensure CORS is enabled (handled automatically)

4. **Slow Generation**
   - First run downloads the model (one-time setup)
   - Subsequent runs should be much faster
   - GPU/MPS acceleration is used automatically when available

5. **Memory Issues**
   - The quantized model uses ~200MB RAM
   - Close other applications if needed
   - Restart API server if generation becomes slow

**Browser compatibility:**
- Works best in Chrome, Firefox, Safari, Edge
- Requires JavaScript enabled

### Performance Tips

1. **First Run**: Initial model download takes time but only happens once
2. **Memory**: Keep other applications closed for best performance  
3. **Generation**: Click panels strategically - each generates new content
4. **Browser**: Use hardware acceleration for smoother interactions
5. **API**: Keep the server running to avoid restart delays

## Contributing

We welcome contributions! Areas for improvement:

- Additional language models
- Visual themes and styles (Much needed)
- Collaborative editing features

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Hugging Face for the GPT-2 model
- Panzoom.js for smooth zoom interactions
- The fractal mathematics community for inspiration

---

*"Every story contains infinite stories. Every zoom reveals new worlds."*