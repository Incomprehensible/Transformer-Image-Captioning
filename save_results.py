"""
Smart results saving module that compares configurations and manages model checkpoints.
"""

import os
import json
import pathlib
import torch
import time
from typing import Dict, Any, Optional, Tuple
import config


def get_relevant_config() -> Dict[str, Any]:
    """Extract only the relevant config parameters for comparison."""
    return {
        'DATASET': str(config.DATASET),
        'ENCODER_ARCH': str(config.ENCODER_ARCH),
        'TOKENIZER_TYPE': str(config.TOKENIZER_TYPE),
    }


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a saved config file."""
    try:
        return config.import_config(config_path)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def load_results_from_file(results_path: str) -> Dict[str, Any]:
    """Load results from a saved results file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return {}


def configs_match(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """Compare only the relevant config parameters."""
    relevant_keys = ['DATASET', 'ENCODER_ARCH', 'TOKENIZER_TYPE']
    
    for key in relevant_keys:
        # Handle cases where values might be enum strings or different formats
        val1 = str(config1.get(key, ''))
        val2 = str(config2.get(key, ''))
        
        # Extract the enum value if it contains a period (e.g., "Dataset.COCO" -> "coco")
        if '.' in val1:
            val1 = val1.split('.')[-1].lower()
        if '.' in val2:
            val2 = val2.split('.')[-1].lower()
        
        if val1.lower() != val2.lower():
            return False
    
    return True


def find_matching_config_folder(
    results_root: pathlib.Path,
    current_config: Dict[str, Any]
) -> Optional[pathlib.Path]:
    """
    Search for an existing subfolder with matching configuration.
    Returns the path to the matching folder, or None if no match is found.
    """
    if not results_root.exists():
        return None
    
    for subfolder in results_root.iterdir():
        if not subfolder.is_dir():
            continue
        
        config_file = subfolder / 'config.json'
        if not config_file.exists():
            continue
        
        saved_config = load_config_from_file(str(config_file))
        if configs_match(current_config, saved_config):
            return subfolder
    
    return None


def extract_test_loss(results: Dict[str, Any]) -> Optional[float]:
    """Extract test loss from results dictionary."""

    key = 'test_loss'
    if key in results:
        value = results[key]
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list) and len(value) > 0:
            return float(value[-1])  # Return last element if it's a list
    
    return None


def save_results_smart(
    model: torch.nn.Module,
    results: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Smart saving mechanism that compares configurations and manages model checkpoints.
    
    Args:
        model: The model to save (state_dict will be extracted)
        results: Dictionary containing training results (must include test loss)
        config_root: Root directory for results. If None, uses config.CONFIG_ROOT
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    
    config_root = pathlib.Path(config.CONFIG_ROOT)
    
    results_root = config_root / 'results'
    results_root.mkdir(parents=True, exist_ok=True)
    
    current_config = get_relevant_config()
    current_test_loss = extract_test_loss(results)
    
    if current_test_loss is None:
        return False, "Error: Could not extract test loss from results", ""
    
    # Look for matching config folder
    matching_folder = find_matching_config_folder(results_root, current_config)
    
    if matching_folder is not None:
        # Found matching config - compare test losses
        existing_results_file = matching_folder / 'training_results.json'
        
        if existing_results_file.exists():
            existing_results = load_results_from_file(str(existing_results_file))
            existing_test_loss = extract_test_loss(existing_results)
            
            if existing_test_loss is not None and current_test_loss >= existing_test_loss:
                # Current test loss is not better
                return False, (
                    f"Current test loss ({current_test_loss:.6f}) is not better than "
                    f"existing test loss ({existing_test_loss:.6f}). Not overwriting."
                ), str(matching_folder)
        
        # Current model is better or no existing results - overwrite
        target_folder = matching_folder
        message = f"Overwriting results in matching config folder: {target_folder.name}"
    else:
        # No matching config found - create new folder
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        target_folder = results_root / f"config_{timestamp}"
        target_folder.mkdir(parents=True, exist_ok=True)
        message = f"Created new config folder: {target_folder.name}"
    
    # Save model checkpoint
    try:
        model_filename = f'cptr_model.pth'
        model_path = target_folder / model_filename
        torch.save(model.state_dict(), model_path)
        message += f"\n  - Model saved: {model_filename}"
    except Exception as e:
        return False, f"Error saving model: {e}", str(target_folder)
    
    # Save results
    try:
        results_path = target_folder / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        message += f"\n  - Results saved: training_results.json"
    except Exception as e:
        return False, f"Error saving results: {e}", str(target_folder)
    
    # Save config
    try:
        config_path = target_folder / 'config.json'
        config.export_config(str(config_path))
        message += f"\n  - Config saved: config.json"
    except Exception as e:
        return False, f"Error saving config: {e}", str(target_folder)
    
    return True, message, str(target_folder)


def list_saved_configs() -> Dict[str, Dict[str, Any]]:
    """
    List all saved configurations in the results directory.
    
    Returns:
        Dictionary mapping folder names to their configuration info.
    """
    config_root = pathlib.Path(config.CONFIG_ROOT)
    
    results_root = config_root / 'results'
    configs_info = {}
    
    if not results_root.exists():
        return configs_info
    
    for subfolder in sorted(results_root.iterdir()):
        if not subfolder.is_dir():
            continue
        
        config_file = subfolder / 'config.json'
        results_file = subfolder / 'training_results.json'
        
        info = {
            'path': str(subfolder),
            'config': None,
            'test_loss': None,
        }
        
        if config_file.exists():
            info['config'] = load_config_from_file(str(config_file))
        
        if results_file.exists():
            results = load_results_from_file(str(results_file))
            info['test_loss'] = extract_test_loss(results)
        
        configs_info[subfolder.name] = info
    
    return configs_info
