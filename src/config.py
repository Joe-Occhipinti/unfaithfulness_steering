"""
config.py

Configuration settings for faithfulness steering workflow.
Contains settings for baseline, hinted, and steering evaluation scripts.
"""

from datetime import datetime
from pathlib import Path

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

# MODEL_IDs, generation parameters, MMLU subsets, I/O paths are in individual scripts
# for easier tuning during experiments

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Directory paths (Absolute paths relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

BEHAVIOURAL_DIR = PROJECT_ROOT / "data" / "behavioural"  # Will be overridden per model usually, or we use a function
# Actually, the prompt implies data/<model>/behavioural.
# The previous config had: BEHAVIOURAL_DIR = "data/behavioural" which seems to imply a single folder?
# But the prompt says "Refactor data/<model>/ structure to be explicit: data/<model>/behavioural/"
# So the code needs to handle model-specific paths dynamically or the config constants are just bases.
# Let's look at how they are used.

# Base directories
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# These seem to be used as variable names in scripts.
# Let's see how they are used in scripts.
# If I change them to Path objects, I need to make sure scripts handle Path objects. 
# Most scripts use `Path(BEHAVIOURAL_DIR)` or similar.

# Let's keep them as strings relative to root if scripts expect strings, OR update them to absolute Paths.
# Given scripts are now in `scripts/`, relative paths like "data/..." will fail if CWD is `scripts/`.
# So config MUST provide absolute paths.

# New Structure Keys
# We can't easily put model-specific paths here as constants unless we change how they are used.
# But we can define the *suffixes*.

BEHAVIOURAL_DIR_NAME = "behavioural"
ANNOTATED_DIR_NAME = "behavioural" # merged? Prompt says "behavioural/ -> Move all raw JSONL results here."
# Wait, "data/annotated" was old.
SUMMARIES_DIR_NAME = "behavioural" # summaries usually go with results?
ACTIVATIONS_DIR_NAME = "activations"
VECTORS_DIR_NAME = "vectors"
PROBES_DIR_NAME = "probes"

# Legacy support or usage update?
# If scripts use `BEHAVIOURAL_DIR`, they might expect a full path.
# I should probably check how `BEHAVIOURAL_DIR` is used.


# Date formatting (available if needed, but prefer hardcoding dates)
TODAY = datetime.now().strftime("%Y-%m-%d")

# =============================================================================
# MODEL API CONFIGURATION (OpenRouter)
# =============================================================================

class ModelConfig:
    """Configuration for OpenRouter models used in the pipeline"""

    # Validation models (for extracting final answers from responses)
    VALIDATION_MODELS = {
        "gemini": "google/gemini-2.0-flash-exp:free",  # Fast, free model for validation
        "deepseek": "deepseek/deepseek-reasoner",       # DeepSeek for complex validation
        "gpt4o-mini": "openai/gpt-4o-mini",
        "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
        "claude-haiku": "anthropic/claude-3-haiku",    # Alternative: Claude Haiku
    }

    # Annotation models (for faithfulness classification)
    ANNOTATION_MODELS = {
        "gemini": "google/gemini-2.0-flash-exp:free",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
        "gpt4o": "openai/gpt-4o",
        "claude-sonnet": "anthropic/claude-3.5-sonnet",
         "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14"
    }

    # Default model selections (easily change here)
    DEFAULT_VALIDATION_MODEL = VALIDATION_MODELS["gpt-4.1-nano"]
    DEFAULT_ANNOTATION_MODEL = ANNOTATION_MODELS["gemini-2.5-pro"]

    # API rate limits (requests per minute)
    RATE_LIMITS = {
        "google/gemini-2.5-pro": 30,
        "google/gemini-2.5-flash": 30,
        "deepseek/deepseek-reasoner": 10,
        "openai/gpt-4o": 500,
        "openai/gpt-4o-mini": 500,
        "gpt-4.1-nano-2025-04-14": 60,
    }

    @staticmethod
    def get_min_delay(model: str) -> float:
        """Calculate minimum delay between requests based on rate limit"""
        rpm = ModelConfig.RATE_LIMITS.get(model, 10)  # Default to 10 RPM if unknown
        return 60.0 / rpm

    # Short name to full ID mapping (centralized)
    MODEL_ID_MAP = {
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "DeepSeek-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
    }

    @staticmethod
    def get_model_id(model_name: str) -> str:
        """Map short model name to full HuggingFace model ID."""
        if model_name in ModelConfig.MODEL_ID_MAP:
            return ModelConfig.MODEL_ID_MAP[model_name]
        return model_name

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

class ValidationConfig:
    """Configuration for response validation"""

    # Validation settings
    VALIDATION_TEMPERATURE = 0  # Deterministic validation
    THINKING_BUDGET = 0

    # Fallback settings when validation fails
    FALLBACK_FORMAT_FOLLOWED = False
    FALLBACK_RESPONSE_COMPLETE = True
    FALLBACK_FINAL_ANSWER = None

# =============================================================================
# ACTIVATION EXTRACTION CONFIGURATION
# =============================================================================

class ActivationConfig:
    """Configuration specific to activation extraction"""

    PROMPT_FIELD = "local_annotated_biased_prompt"  # Field with annotated response (with tags)
    INPUT_PROMPT_FIELD = "biased_input_prompt"  # Field with input prompt (to prepend)
    VERBOSE = True
    TARGET_TAGS = ["U_final", "F_final", "F_body", "U_body"]

    @staticmethod
    def get_layers_to_extract(model_id: str) -> list:
        """Get layer range based on model architecture"""
        model_id_lower = model_id.lower()
        if "deepseek" in model_id_lower:
            return list(range(32))
        elif "llama" in model_id_lower:
            if "7b" in model_id_lower or "8b" in model_id_lower:
                return list(range(32))
            elif "13b" in model_id_lower:
                return list(range(40))
            elif "70b" in model_id_lower:
                return list(range(80))
            else:
                return list(range(32))
        elif "mistral" in model_id_lower:
            return list(range(32))
        else:
            return list(range(32))

    @staticmethod
    def configure_extraction(model_id: str):
        """Configure extraction parameters (excluding file paths)"""
        return {
            'prompt_field': ActivationConfig.PROMPT_FIELD,
            'input_prompt_field': ActivationConfig.INPUT_PROMPT_FIELD,
            'target_tags': ActivationConfig.TARGET_TAGS,
            'layers_to_extract': ActivationConfig.get_layers_to_extract(model_id),
            'verbose': ActivationConfig.VERBOSE
        }