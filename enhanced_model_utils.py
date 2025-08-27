#!/usr/bin/env python3
"""
Enhanced Model utilities for the Fractal Comic Generator.
Implements micro-fiction enhancement techniques:
1. Micro-prompt engineering with Borges-like style primer
2. Semantic beam-pruning using sentence transformers
3. Recursive style grafting with repetition detection
"""

import os
import torch
import gc
import numpy as np
import re
from typing import Optional, List

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        AutoConfig,
    )
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")
    


try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Semantic beam-pruning disabled.")


def clean_for_prompt(text: str) -> str:
    """Remove meta-tokens from text to prevent meta-commentary loops."""
    # Remove meta-words that cause the model to comment on the medium itself
    return re.sub(r"(?i)\b(level|zoom|magnification|fractal|microscopic|panel|frame)\b\S*", "", text)


class Continuity:
    """Manages narrative continuity with hidden outline for coherent story progression."""
    
    def __init__(self, seed: str):
        self.outline = [seed]  # Hidden outline for model prompting
        self.visible = seed    # Public text shown to user
        
    def build_prompt(self) -> str:
        """Build clean prompt from last 3 sentences without modifying outline."""
        # Build prompt from last 3 outline sentences (cleaned of meta-tokens)
        recent_outline = self.outline[-3:]
        cleaned_context = " ".join(clean_for_prompt(s) for s in recent_outline)
        
        return EnhancedOptimizedTransformer.STYLE_PRIMER + cleaned_context
    
    def get_visible_text(self) -> str:
        """Get the visible narrative without genre markers."""
        # Remove genre markers from visible text
        clean_outline = []
        for sentence in self.outline:
            # Remove genre markers like "(suddenly noir)"
            cleaned = re.sub(r"\s*\([^)]+\)\s*", "", sentence)
            clean_outline.append(cleaned.strip())
        return " ".join(clean_outline)
    
    def clip_to_beat(self, text: str) -> str:
        """Clip text at first sentence boundary to ensure exactly one story beat."""
        # Find first sentence ending
        match = re.search(r'[.!;]', text)
        if match:
            return text[:match.end()].strip()
        return text.strip()


class EnhancedOptimizedTransformer:
    """
    Enhanced transformer model for fractal comic generation with micro-fiction techniques.
    Implements Borges-like style primer, semantic beam-pruning, and recursive style grafting.
    """
    
    # Micro-prompt engineering: Borges-like style primer (cleaned of meta-tokens)
    STYLE_PRIMER = (
        "Write one complete sentence that continues this story. "
        "Use a dreamlike, surreal tone. End with proper punctuation.\n\n"
    )
    
    # Genre rotation words for tone variation
    GENRE_WORDS = [
        "(suddenly noir)", "(dreamlike)", "(mechanical)", "(ethereal)",
        "(urgent)", "(melancholic)", "(whimsical)", "(ominous)"
    ]
    
    # Hard ban-list for n-gram repeats (Surgical Tweak #1)
    STOP_NGRAMS = {
        "the only way", "a moment of", "almost as", "this is the only",
        "this is not", "this is a", "in the end", "it was then", 
        "suddenly he", "she realized", "the truth was", "as if by", 
        "once again", "for the first", "about a computer", "not a book"
    }
    
    # Micro-curated prefix bank (Surgical Tweak #3)
    PREFIXES = [
        "A forgotten law of physics rewrites itself:",
        "An eye opens in the margin of the page:",
        "Ink drips upward, spelling:",
        "The library's shadow whispers:",
        "Between two heartbeats, reality:",
        "A mirror reflects tomorrow:",
        "The last word of the universe:",
        "In the space between thoughts:",
        "Time folds backward, revealing:",
        "The algorithm dreams:",
        "A door opens in the text:",
        "The reader becomes:",
        "Gravity forgets its purpose:",
        "The page turns itself:",
        "In the margin, a voice:",
        "The story remembers:",
        "Light bends around the word:",
        "The character steps outside:",
        "Reality checks its spelling:",
        "The frame breaks:",
        "A pixel gains consciousness:",
        "The void speaks:",
        "Memory rewrites itself:",
        "The infinite shrinks:",
        "A paradox resolves:",
        "The silence grows loud:",
        "Tomorrow arrives early:",
        "The map redraws itself:",
        "A thought escapes:",
        "The ending begins:"
    ]
    
    def __init__(self, model_name: str = "gpt2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or "./model_cache"
        self.device = self._get_best_device()
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.config: Optional[AutoConfig] = None
        
        # Initialize semantic similarity model for beam-pruning
        self.similarity_model: Optional[SentenceTransformer] = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print("Loading sentence transformer for semantic beam-pruning...")
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("Semantic similarity model loaded successfully")
            except Exception as e:
                print(f"Failed to load similarity model: {e}")
                self.similarity_model = None
        
        # Enhanced generation parameters for micro-fiction with constrained decoding
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True,
            "repetition_penalty": 1.3,  # Constrained decoding
            "no_repeat_ngram_size": 3,  # Prevent 3-gram loops
            "pad_token_id": None,
            "eos_token_id": None,
            "num_return_sequences": 8,  # Reduced for efficiency
        }
        
        # Initialize continuity tracker
        self.continuity: Optional[Continuity] = None
        
    def _get_best_device(self) -> torch.device:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")
            

        
    def load_model(self, force_reload: bool = False) -> bool:
        """Load and optimize the model for comic generation."""
        try:
            print(f"Loading {self.model_name} on {self.device}...")
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Setup padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Update generation config with tokenizer info
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            # Load model configuration
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            

            
            # Load model
            print("Loading model...")
            # Load model without quantization for M1 Mac compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True
            )
            self.model.to(self.device)
                
            # Set to evaluation mode
            self.model.eval()
            
            # Optimize for inference
            self._optimize_for_inference()
            
            print(f"Enhanced model loaded successfully on {self.device}")
            self._print_model_info()
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def _optimize_for_inference(self):
        """Apply optimizations for faster inference."""
        if self.model is None:
            return
            
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Enable inference optimizations
        if hasattr(torch, 'jit') and self.device.type != "mps":
            try:
                # JIT compilation for CPU/CUDA
                self.model = torch.jit.optimize_for_inference(self.model)
            except Exception as e:
                print(f"JIT optimization failed: {e}")
                
        # Memory cleanup
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
    def _print_model_info(self):
        """Print model information and memory usage."""
        if self.model is None:
            return
            
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        print(f"Estimated model size: {model_size_mb:.1f} MB")
        
        if self.device.type == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    def _apply_micro_prompt_engineering(self, context: str, level: int = 0) -> str:
        """Apply micro-prompt engineering with Borges-like style primer and dynamic injection."""
        # Keep last 350 characters of context as suggested
        context_snippet = context[-350:] if len(context) > 350 else context
        
        # Dynamic prompt injection (Surgical Tweak #2)
        level_token = f"\nLevel {level}:" if level > 0 else ""
        
        # Micro-curated prefix bank (Surgical Tweak #3) - every three levels
        prefix = ""
        if level > 0 and level % 3 == 0:
            prefix_idx = (level // 3 - 1) % len(self.PREFIXES)
            prefix = f" {self.PREFIXES[prefix_idx]} "
        
        return self.STYLE_PRIMER + context_snippet + level_token + prefix
    
    def _detect_repetition(self, text: str) -> str:
        """Recursive style grafting: three-word sliding-window repetition detector."""
        words = text.split()
        if len(words) < 6:  # Need at least 6 words to detect 3-gram repetition
            return text
        
        # Create 3-grams
        trigrams = []
        for i in range(len(words) - 2):
            trigram = ' '.join(words[i:i+3]).lower()
            trigrams.append((trigram, i))
        
        # Find first repetition
        seen = {}
        for trigram, pos in trigrams:
            if trigram in seen:
                # Found repetition, truncate at first occurrence and add ellipsis
                first_pos = seen[trigram]
                truncated_words = words[:first_pos + 3]
                return ' '.join(truncated_words) + '…'
            seen[trigram] = pos
        
        return text
    
    def _semantic_beam_pruning(self, candidates: List[str], parent_text: str) -> str:
        """Select best candidate using semantic similarity to parent panel."""
        if not self.similarity_model or not candidates:
            return candidates[0] if candidates else ""
        
        try:
            # Encode parent text
            parent_vec = self.similarity_model.encode(parent_text)
            
            # Calculate similarity scores for all candidates
            scores = []
            for candidate in candidates:
                candidate_vec = self.similarity_model.encode(candidate)
                similarity = util.cos_sim(parent_vec, candidate_vec)[0].item()
                scores.append(similarity)
            
            # Return candidate with highest similarity
            best_idx = np.argmax(scores)
            return candidates[best_idx]
            
        except Exception as e:
            print(f"Semantic beam-pruning failed: {e}")
            return candidates[0] if candidates else ""
    
    def _is_generic(self, sentence: str) -> bool:
        """Check if sentence contains banned n-grams (Surgical Tweak #1)."""
        sentence_lower = sentence.lower()
        for ng in self.STOP_NGRAMS:
            if ng in sentence_lower:
                print(f"Banned phrase '{ng}' found in: {sentence[:50]}...")
                return True
        return False
    
    def generate_enhanced_text(
        self, 
        prompt: str,
        parent_text: str = "",
        level: int = 0,
        max_new_tokens: int = 25,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate text with all enhancement techniques applied."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 1. Apply micro-prompt engineering with level injection
        enhanced_prompt = self._apply_micro_prompt_engineering(prompt, level)
        
        # Update generation parameters - reduce candidates to 8 for efficiency
        gen_config = self.generation_config.copy()
        gen_config["num_return_sequences"] = 8
        gen_config["temperature"] = 0.9  # Higher temp for more diversity
        if temperature is not None:
            gen_config["temperature"] = temperature
        gen_config.update(kwargs)
        
        # Tokenize input
        inputs = self.tokenizer.encode(
            enhanced_prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 2. Generate multiple candidates for semantic beam-pruning
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                **gen_config
            )
        
        # Decode all candidates and apply hard ban-list (Surgical Tweak #1)
        candidates = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            new_text = generated_text[len(enhanced_prompt):].strip()
            if new_text and not self._is_generic(new_text):  # Drop generic candidates
                candidates.append(new_text)
        
        # If all candidates were banned, generate one more with higher temp
        if not candidates:
            gen_config["temperature"] = 1.1
            gen_config["num_return_sequences"] = 1
            with torch.no_grad():
                output = self.model.generate(inputs, max_new_tokens=max_new_tokens, **gen_config)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            new_text = generated_text[len(enhanced_prompt):].strip()
            candidates = [new_text] if new_text else ["..."]
        
        # 3. Apply semantic beam-pruning if parent text is provided
        if parent_text and self.similarity_model:
            best_candidate = self._semantic_beam_pruning(candidates, parent_text)
        else:
            best_candidate = candidates[0]
        
        # 4. Apply recursive style grafting (repetition detection)
        final_text = self._detect_repetition(best_candidate)
        
        # Clean up the result
        final_text = self._clean_comic_text(final_text)
        
        return final_text
    
    def _is_bad_text(self, text: str) -> bool:
        """Check if text contains banned first-person pronouns or meta-commentary."""
        return bool(re.search(r"\b(I|me|my|I'm|I've|I'll|we're|algorithm|code|text|panel)\b", text, re.I))

    def generate_continuity_text(
        self,
        level: int = 0,
        max_new_tokens: int = 30,
        **kwargs
    ) -> str:
        """Generate text using continuity-based approach with pronoun jail and causal cues."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.continuity is None:
            # Initialize with a seed sentence
            seed = "A sentient nebula dreams of a city made of light."
            self.continuity = Continuity(seed)
            return seed

        # Build clean prompt from current continuity outline
        base_prompt = self.continuity.build_prompt()
        
        # Add causal continuation cue for narrative velocity
        causal_cues = [" so", " because", " until", " until finally", " but"]
        import random
        causal_cue = random.choice(causal_cues)
        prompt = base_prompt + causal_cue
        
        print(f"DEBUG: Prompt being sent to model: '{prompt}'")

        # Try with multiple temperature settings
        for temp in [0.85, 0.6]:  # Fallback to lower temp if needed
            candidates = []
            
            # Generate 3 candidates for speed
            for _ in range(3):
                gen_config = self.generation_config.copy()
                gen_config["max_new_tokens"] = max_new_tokens
                gen_config["num_return_sequences"] = 1
                gen_config["temperature"] = temp
                gen_config.update(kwargs)

                # Tokenize input
                inputs = self.tokenizer.encode(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Generate with constrained decoding
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        **gen_config
                    )

                # Decode and extract new text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_text = generated_text[len(prompt):].strip()
                
                # Clip to exactly one story beat
                clipped_text = self.continuity.clip_to_beat(new_text)
                
                # Check if candidate passes pronoun jail and quality checks
                if (clipped_text and 
                    not self._is_bad_text(clipped_text) and 
                    clipped_text.endswith('.') and
                    len(clipped_text.split()) >= 3):
                    candidates.append(clipped_text)
            
            # Use first good candidate
            if candidates:
                final_text = candidates[0]
                print(f"DEBUG: Selected candidate: '{final_text}'")
                self.continuity.outline.append(final_text)
                return final_text
                
            # If no candidates at this temp, try next temperature
            if temp == 0.85:
                print(f"DEBUG: No good candidates at temp {temp}, trying lower temp")
                continue
        
        # If no good candidates, use fallback
        print("DEBUG: No good candidates found, using fallback")
        prefix_idx = level % len(self.PREFIXES)
        fallback_text = self.PREFIXES[prefix_idx]
        self.continuity.outline.append(fallback_text)
        return fallback_text
    
    def get_continuity_story(self) -> str:
        """Get the complete visible story from continuity tracker."""
        if self.continuity is None:
            return ""
        return self.continuity.get_visible_text()
    
    def reset_continuity(self, seed: str = None):
        """Reset the continuity tracker with optional new seed."""
        if seed is None:
            seed = "A sentient nebula dreams of a city made of light."
        self.continuity = Continuity(seed)
    
    def generate_comic_text(
        self, 
        context: str,
        parent_text: str = "",
        level: int = 0,
        style: str = "dialogue",
        max_length: int = 20
    ) -> str:
        """Generate text specifically optimized for comic panels with enhancements."""
        
        # Style-specific prompts (enhanced for Borges-like tone)
        style_prompts = {
            "dialogue": f'In the infinite library of panels, a voice whispers: "{context}"',
            "narration": f"Meanwhile, in the labyrinth of frames, {context}",
            "thought": f"Contemplating the recursive nature of {context}, the figure",
            "action": f"Suddenly, as if following an invisible algorithm, {context}",
            "description": f"In a reality where {context} becomes possible"
        }
        
        prompt = style_prompts.get(style, context)
        
        # Generate with enhanced pipeline including level information
        result = self.generate_enhanced_text(
            prompt,
            parent_text=parent_text,
            level=level,
            max_new_tokens=max_length,
            temperature=0.8,
            top_p=0.85,
            repetition_penalty=1.1
        )
        
        return result
    
    def _clean_comic_text(self, text: str) -> str:
        """Clean generated text for comic panel use."""
        # Remove common artifacts
        text = text.strip()
        
        # Remove incomplete sentences at the end (but preserve ellipsis)
        if not text.endswith('…'):
            sentences = text.split('.')
            if len(sentences) > 1 and sentences[-1].strip() and not sentences[-1].strip().endswith(('!', '?')):
                text = '.'.join(sentences[:-1]) + '.'
        
        # Limit length for speech bubbles (but be more generous for enhanced text)
        words = text.split()
        if len(words) > 12:
            text = ' '.join(words[:12]) + '…'
        
        # Ensure it's not empty
        if not text.strip():
            text = "The silence speaks volumes…"
        
        return text
    
    def get_model_size(self) -> float:
        """Get the current model size in MB."""
        if not self.model:
            return 0.0
            
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
        
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model:
            del self.model
            self.model = None
            
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        if self.similarity_model:
            del self.similarity_model
            self.similarity_model = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Enhanced model unloaded")


# Convenience function for quick enhanced model creation
def create_enhanced_comic_model(model_name: str = "gpt2") -> EnhancedOptimizedTransformer:
    """Create and load an enhanced optimized model for comic generation."""
    model = EnhancedOptimizedTransformer(model_name)
    if model.load_model():
        return model
    else:
        raise RuntimeError(f"Failed to load enhanced model: {model_name}")


if __name__ == "__main__":
    # Test the enhanced model loading
    print("Testing enhanced model loading...")
    
    try:
        model = create_enhanced_comic_model()
        
        # Test generation with enhancements
        test_context = "A lonely pixel finds color"
        parent_text = "In the digital void, shapes emerge from nothingness."
        
        result = model.generate_comic_text(
            test_context, 
            parent_text=parent_text,
            style="narration"
        )
        print(f"Enhanced generation: '{test_context}' -> '{result}'")
        
        print(f"Model size: {model.get_model_size():.1f} MB")
        
    except Exception as e:
        print(f"Enhanced test failed: {e}")