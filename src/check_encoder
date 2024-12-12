import torch
import argparse
import os

def load_state_dict(state_dict_path):
    """
    Load the state_dict from the given path.

    Args:
        state_dict_path (str): Path to the state_dict file.

    Returns:
        dict: Loaded state_dict.
    """
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"The state_dict file was not found at: {state_dict_path}")
    
    try:
        state_dict = torch.load(state_dict_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Error loading state_dict: {e}")
    
    if not isinstance(state_dict, dict):
        raise ValueError("The loaded state_dict is not a dictionary.")
    
    return state_dict

def print_state_dict_info(state_dict):
    """
    Print information about the state_dict.

    Args:
        state_dict (dict): The state_dict to inspect.
    """
    print(f"Total parameters and buffers in state_dict: {len(state_dict)}\n")
    for key in state_dict:
        param = state_dict[key]
        print(f"Key: {key}")
        print(f" - Shape: {param.shape}")
        print(f" - Requires Grad: {param.requires_grad}\n")

def check_expected_keys(state_dict, expected_keys):
    """
    Check if the expected keys are present in the state_dict.

    Args:
        state_dict (dict): The state_dict to inspect.
        expected_keys (list): List of expected key names.

    Returns:
        list: Missing keys.
    """
    missing_keys = [key for key in expected_keys if key not in state_dict]
    return missing_keys

def main():
    parser = argparse.ArgumentParser(description="Inspect a trained encoder's state_dict.")
    parser.add_argument('--state_dict_path', type=str, required=True, 
                        help='Path to the trained encoder state_dict file (e.g., trained_encoder_state_dict.pt).')
    parser.add_argument('--expected_keys_file', type=str, default=None,
                        help='Optional path to a text file containing expected keys, one per line.')
    args = parser.parse_args()

    # Load the state_dict
    state_dict = load_state_dict(args.state_dict_path)
    print(f"Loaded state_dict from: {args.state_dict_path}\n")

    # Print state_dict information
    print_state_dict_info(state_dict)

    # Optional: Check for expected keys
    if args.expected_keys_file:
        if not os.path.isfile(args.expected_keys_file):
            print(f"Expected keys file not found at: {args.expected_keys_file}")
        else:
            with open(args.expected_keys_file, 'r') as f:
                expected_keys = [line.strip() for line in f if line.strip()]
            missing_keys = check_expected_keys(state_dict, expected_keys)
            if missing_keys:
                print("Missing expected keys in the state_dict:")
                for key in missing_keys:
                    print(f" - {key}")
            else:
                print("All expected keys are present in the state_dict.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:36:27 2024

@author: ellie
"""

