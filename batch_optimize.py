"""
Batch prompt optimization script
Optimizes prompts using multiple LLM models (Qwen API and Ollama local models)
Supports both single system prompt and dual-variant system prompts (no_opt/opt)
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# ============================================================================
# Configuration
# ============================================================================

# Input prompts file (one prompt per line)
PROMPTS_FILE = "null_350.txt"

# System prompt file (optional, for dual-variant mode)
# If no optimization variant: import "ablated system prompt.txt"
# Otherwise: import "full system prompt.txt"
SYSTEM_FILE = "system_prompt.txt"

# Output directory
OUTPUT_ROOT = "optimization_results"

# Default system prompt (used when SYSTEM_FILE is not provided)
DEFAULT_SYSTEM_PROMPT = """
Optimize user prompts
[Output Restrictions]
The final output must contain only one prompt text [MUST COMPLY].
No analysis, explanation, preface, summary, comments or any extra content is allowed.
If the output contains any text other than the prompt, it is considered a failure.
Please respond in Chinese.
"""

# Model configurations
# Format: (model_name, model_type)
# model_type: "qwen" for Qwen API, "ollama" for Ollama local models
DEFAULT_MODELS = [
    ("llama3:8b", "ollama"),
    ("qwen-max", "qwen"),
    ("qwen3:4b", "ollama"),
]

# API Configuration (set via environment variables)
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")  # Set via environment variable
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2
TIMEOUT_SECONDS = 120


# ============================================================================
# Helper Functions
# ============================================================================

def safe_filename(name: str) -> str:
    """
    Convert model name to Windows-safe filename.
    Replaces special characters that are not allowed in Windows filenames.
    """
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def load_prompts(file_path: str) -> List[str]:
    """
    Load prompts from a text file (one prompt per line).
    
    Args:
        file_path: Path to the prompts file
        
    Returns:
        List of prompt strings
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")
    
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                prompts.append(line)
    return prompts


def load_system_variants(file_path: str) -> Optional[Dict[str, str]]:
    """
    Load system prompt variants from file.
    Expected format: "不加优化：... 加优化：..."
    
    Args:
        file_path: Path to the system prompt file
        
    Returns:
        Dictionary with "no_opt" and "opt" keys, or None if file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        return None
    
    raw = path.read_text(encoding="utf-8")
    # Match pattern: "不加优化：... 加优化：..."
    match = re.search(r"不加优化：\s*(.*?)\s*加优化：\s*(.*)\s*$", raw, re.S)
    if not match:
        return None
    
    return {
        "no_opt": match.group(1).strip(),
        "opt": match.group(2).strip(),
    }


# ============================================================================
# LLM Creation
# ============================================================================

def create_llm(model_name: str, model_type: str = "qwen") -> ChatOpenAI:
    """
    Create LLM instance based on model type.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ("qwen" or "ollama")
        
    Returns:
        ChatOpenAI instance
        
    Raises:
        ValueError: If model_type is unknown
    """
    if model_type == "qwen":
        if not QWEN_API_KEY:
            raise ValueError("QWEN_API_KEY environment variable is not set")
        return ChatOpenAI(
            openai_api_key=QWEN_API_KEY,
            openai_api_base=QWEN_API_BASE,
            model_name=model_name,
            temperature=0.1
        )
    elif model_type == "ollama":
        # Ollama local model configuration
        return ChatOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't require a real API key
            model_name=model_name,
            temperature=0.1,
            timeout=TIMEOUT_SECONDS,
            max_retries=2  # Client-level retry
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Prompt Optimization
# ============================================================================

def optimize_prompt_with_model(
    prompt: str,
    llm: ChatOpenAI,
    model_name: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY
) -> str:
    """
    Optimize a prompt using the specified model with retry mechanism.
    
    Args:
        prompt: The prompt to optimize
        llm: ChatOpenAI instance
        model_name: Name of the model (for logging)
        system_prompt: System prompt to use
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (seconds)
        
    Returns:
        Optimized prompt string
    """
    print(f"[{model_name}] Optimizing prompt: {prompt[:50]}...")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please optimize the following prompt:\n{prompt}\nRespond in Chinese")
    ]
    
    # Retry mechanism
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            
            if hasattr(response, 'content'):
                optimized = response.content.strip()
            elif isinstance(response, str):
                optimized = response.strip()
            else:
                optimized = str(response).strip()
            
            print(f"[{model_name}] Optimization completed, result length: {len(optimized)}")
            return optimized
            
        except Exception as e:
            error_str = str(e)
            error_code = None
            
            # Check for server errors (502, 503, 504) or timeout
            if "502" in error_str or "Bad Gateway" in error_str:
                error_code = "502"
            elif "503" in error_str or "Service Unavailable" in error_str:
                error_code = "503"
            elif "504" in error_str or "Gateway Timeout" in error_str:
                error_code = "504"
            elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                error_code = "TIMEOUT"
            elif "connection" in error_str.lower() or "temporarily unavailable" in error_str.lower():
                error_code = "CONNECTION"
            
            # Retry if it's a server error and we have retries left
            if error_code and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                print(f"[{model_name}] Encountered {error_code} error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                # Last attempt or non-server error, return error message
                import traceback
                error_msg = f"[{model_name}] Error during optimization: {error_str}"
                print(error_msg)
                if attempt == max_retries - 1:
                    print(f"[{model_name}] Reached maximum retry count ({max_retries}), giving up")
                print(traceback.format_exc())
                return f"Optimization failed: {error_str}"
    
    # Should not reach here, but for safety
    return "Optimization failed: Unknown error"


# ============================================================================
# Result Saving
# ============================================================================

def save_single_model_result(
    model_name: str,
    model_results: List[Dict[str, Any]],
    variant_name: Optional[str] = None,
    output_dir: str = OUTPUT_ROOT
) -> None:
    """
    Save optimization results for a single model.
    
    Args:
        model_name: Name of the model
        model_results: List of result dictionaries
        variant_name: Optional variant name (for dual-variant mode)
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model name to safe filename
    safe_model_name = safe_filename(model_name)
    
    # Build filename prefix
    if variant_name:
        prefix = f"{safe_model_name}__{variant_name}"
    else:
        prefix = f"{safe_model_name}_optimized"
    
    # Save as text file
    txt_file = os.path.join(output_dir, f"{prefix}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        if variant_name:
            f.write(f"Variant: {variant_name}\n")
        f.write(f"Optimization Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for idx, item in enumerate(model_results, 1):
            f.write(f"Prompt {idx}:\n")
            f.write(f"Original: {item['original']}\n\n")
            f.write(f"Optimized:\n{item['optimized']}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    # Save as JSON file
    json_file = os.path.join(output_dir, f"{prefix}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "variant_name": variant_name,
            "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "results": model_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Model {model_name} results saved:")
    print(f"  - Text file: {txt_file}")
    print(f"  - JSON file: {json_file}")


def save_all_results(
    results: Dict[str, Any],
    output_dir: str = OUTPUT_ROOT
) -> None:
    """
    Save summary of all model results.
    
    Args:
        results: Dictionary containing all model results
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON format
    json_file = os.path.join(output_dir, f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAll results summary saved to: {json_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function to run batch prompt optimization.
    Supports both single system prompt mode and dual-variant mode.
    """
    # Load prompts
    prompts_file = PROMPTS_FILE
    if not os.path.exists(prompts_file):
        print(f"Error: Prompts file not found: {prompts_file}")
        return
    
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts from {prompts_file}\n")
    
    # Load system prompt variants (if available)
    system_variants = load_system_variants(SYSTEM_FILE)
    use_variants = system_variants is not None
    
    if use_variants:
        print(f"System prompt variants loaded: {list(system_variants.keys())}")
        print(f"Will run both variants for each model\n")
    else:
        print(f"Using default system prompt\n")
    
    # Model configuration
    models = DEFAULT_MODELS
    
    # Storage for all results
    all_results = {}
    
    # Process each model
    for model_name, model_type in models:
        print(f"\n{'='*80}")
        print(f"Starting model: {model_name} ({model_type})")
        print(f"{'='*80}\n")
        
        try:
            # Create LLM instance
            llm = create_llm(model_name, model_type)
            
            if use_variants:
                # Dual-variant mode: run both no_opt and opt variants
                for variant_name, system_prompt in system_variants.items():
                    print(f"\n--- Variant: {variant_name} ---")
                    
                    model_results = []
                    for idx, prompt in enumerate(prompts, 1):
                        print(f"\n[{idx}/{len(prompts)}] ", end="")
                        optimized = optimize_prompt_with_model(
                            prompt, llm, model_name, system_prompt
                        )
                        model_results.append({
                            "index": idx,
                            "original": prompt,
                            "optimized": optimized
                        })
                    
                    # Save results for this variant
                    result_key = f"{model_name}__{variant_name}"
                    all_results[result_key] = model_results
                    print(f"\nModel {model_name} ({variant_name}) completed optimization")
                    print(f"\nSaving results for model {model_name} ({variant_name})...")
                    save_single_model_result(model_name, model_results, variant_name)
            else:
                # Single system prompt mode
                model_results = []
                for idx, prompt in enumerate(prompts, 1):
                    print(f"\n[{idx}/{len(prompts)}] ", end="")
                    optimized = optimize_prompt_with_model(prompt, llm, model_name)
                    model_results.append({
                        "index": idx,
                        "original": prompt,
                        "optimized": optimized
                    })
                
                all_results[model_name] = model_results
                print(f"\nModel {model_name} completed optimization")
                
                # Save results immediately
                print(f"\nSaving results for model {model_name}...")
                save_single_model_result(model_name, model_results)
            
        except Exception as e:
            print(f"\nModel {model_name} optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            if use_variants:
                for variant_name in system_variants.keys():
                    all_results[f"{model_name}__{variant_name}"] = []
            else:
                all_results[model_name] = []
    
    # Save summary of all results
    print(f"\n{'='*80}")
    print("Saving summary of all model results...")
    print(f"{'='*80}\n")
    save_all_results(all_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Optimization Summary")
    print(f"{'='*80}")
    for result_key, results in all_results.items():
        print(f"{result_key}: {len(results)} prompts optimized")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
