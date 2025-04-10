import os
from prepare_finetuning_data import (
    clean_code_for_training,
    clean_code_for_combined,
    extract_functions,
    prepare_dataset
)

def test_file_pair():
    # Read test files
    h_file = "GorgeousCoreRuntime/Public/ObjectVariables/GorgeousObjectVariable.h"
    cpp_file = "GorgeousCoreRuntime/Private/SourceFiles/ObjectVariables/GorgeousObjectVariable.cpp"
    
    with open(h_file, 'r', encoding='utf-8') as f:
        h_content = f.read()
    with open(cpp_file, 'r', encoding='utf-8') as f:
        cpp_content = f.read()
    
    print("Testing header file cleaning...")
    h_clean_training = clean_code_for_training(h_content)
    h_clean_combined = clean_code_for_combined(h_content)
    
    print("\nTraining clean header (first 500 chars):")
    print(h_clean_training[:500])
    print("\nCombined clean header (first 500 chars):")
    print(h_clean_combined[:500])
    
    print("\nTesting cpp file cleaning...")
    cpp_clean_training = clean_code_for_training(cpp_content)
    cpp_clean_combined = clean_code_for_combined(cpp_content)
    
    print("\nTraining clean cpp (first 500 chars):")
    print(cpp_clean_training[:500])
    print("\nCombined clean cpp (first 500 chars):")
    print(cpp_clean_combined[:500])
    
    print("\nExtracting functions from header...")
    h_funcs = extract_functions(h_clean_training, '.h', h_content)
    print(f"Found {len(h_funcs)} functions in header")
    for func in h_funcs[:2]:  # Print first 2 functions
        print(f"\nFunction: {func['name']}")
        print(f"Class: {func['class_name']}")
        print(f"Line: {func['line_number']}")
        print(f"Body snippet: {func['body'][:200]}")
    
    print("\nExtracting functions from cpp...")
    cpp_funcs = extract_functions(cpp_clean_training, '.cpp', cpp_content)
    print(f"Found {len(cpp_funcs)} functions in cpp")
    for func in cpp_funcs[:2]:  # Print first 2 functions
        print(f"\nFunction: {func['name']}")
        print(f"Class: {func['class_name']}")
        print(f"Line: {func['line_number']}")
        print(f"Body snippet: {func['body'][:200]}")

if __name__ == "__main__":
    test_file_pair() 