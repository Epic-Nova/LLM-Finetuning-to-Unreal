import os
import json
import re
import logging
import requests
import argparse
import signal
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from functools import wraps
import errno
import threading
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

class TqdmLoggingHandler(logging.Handler):
    """Redirect logging messages through tqdm.write() so the progress bar stays clean."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
            
class Config:
    def __init__(self):
        self.disable_api = False
        self.allowed_extensions = {".cpp", ".h", ".hpp", ".c", ".cc", ".cs"}
        self.output_dir = os.path.join(os.getcwd(), "TrainingData")
        self.combined_output = "combined_dataset.jsonl"
        self.pretty_output = "pretty_dataset.json"
        self.api_endpoint = "LLM_API_ENDPOINT" 
        self.batch_size = 10
        self.debug = False

config = Config()

logger = logging.getLogger("tqdm_logger")

handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

def remove_code_comments(code: str) -> str:
    """Removes all code comments from the input code."""
    # Remove block comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove single line comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    return code

def clean_code_for_training(code: str) -> str:
    """Completely cleans code for TrainingData/ files - removes ALL comments, headers, includes, pragmas."""
    # Remove block comments (including UFUNCTION and UPROPERTY comments)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove single line comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # Remove includes, pragmas, and other directives
    lines = code.splitlines()
    filtered_lines = []
    for line in lines:
        if not any(pattern in line for pattern in ['#include', '#pragma', 'GENERATED_BODY', 'GENERATED_UCLASS_BODY']):
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def clean_code_for_combined(code: str) -> str:
    """Minimal cleaning for combined files - removes ONLY copyright headers and pragmas."""
    lines = code.splitlines()
    filtered_lines = []
    in_copyright = False
    for line in lines:
        # Skip copyright blocks
        if '/*' in line and any(c in line.lower() for c in ['copyright', '===========']):
            in_copyright = True
            continue
        if in_copyright:
            if '*/' in line:
                in_copyright = False
            continue
        # Skip copyright single lines and pragmas
        if not any(pattern in line.lower() for pattern in ['copyright', '#pragma']):
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def collect_file_paths(path):
    """Collects all allowed file paths from a directory or returns the file if it's a single file."""
    collected_files = []
    
    # Convert to absolute path and normalize
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        logger.error(f"❌ Path does not exist: {path}")
        return collected_files
    
    # If path is a file, check if it's allowed and return it
    if os.path.isfile(abs_path):
        ext = os.path.splitext(abs_path)[1].lower()
        if ext in config.allowed_extensions:
            collected_files.append(abs_path)
            logger.debug(f"🔍 Added file: {abs_path}")
        return collected_files
    
    # If path is a directory, walk through it
    if os.path.isdir(abs_path):
        for root, dirs, files in os.walk(abs_path):
            # Skip common build/intermediate directories
            skip_dirs = ['Intermediate', 'Binaries', 'Saved', '.git', '.vs']
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in config.allowed_extensions:
                    full_path = os.path.join(root, file)
                    collected_files.append(full_path)
                    logger.debug(f"🔍 Found file: {full_path}")
    
    if not collected_files:
        logger.warning(f"⚠️ No matching files found in {path}")
    else:
        logger.debug(f"📁 Collected {len(collected_files)} files from {path}")
    
    return collected_files

def find_line_number(text: str, pattern: str, original_code: str) -> int:
    """Finds the line number of a pattern in the original code."""
    try:
        # Split both into lines for comparison
        orig_lines = original_code.splitlines()
        pattern_lines = pattern.splitlines()
        
        # Try to find the first line of the pattern
        first_pattern_line = pattern_lines[0].strip()
        
        for i, line in enumerate(orig_lines, 1):
            if first_pattern_line in line:
                # Verify this is actually our pattern by checking subsequent lines
                found = True
                for j, pattern_line in enumerate(pattern_lines[1:], 1):
                    if i+j > len(orig_lines) or pattern_line.strip() not in orig_lines[i+j-1]:
                        found = False
                        break
                if found:
                    return i
        
        # If we couldn't find an exact match, try a fuzzy match on the first line
        for i, line in enumerate(orig_lines, 1):
            if any(word in line for word in first_pattern_line.split()):
                return i
    except Exception as e:
        logger.warning(f"⚠️ Error finding line number: {e}")
    
    return 1

def extract_functions(code: str, file_type: str, original_code: str = None) -> List[Dict[str, Any]]:
    """Extracts functions based on file type. Uses original_code for line number mapping."""
    functions = []
    if original_code is None:
        original_code = code

    logger.debug("🔍 extract_functions START")
    start_time = time.time()

    try:
        # Zunächst: Klassen erkennen für Scoped-Funktionen
        class_pattern = r'UCLASS\([^)]*\)\s*class\s+\w+_API\s+(\w+)(?:\s*final)?\s*:\s*(?:public|private|protected)?\s*\w+(?:\s*,\s*(?:public|private|protected)?\s*\w+)*\s*\{([\s\S]*?)\}'
        class_matches = list(re.finditer(class_pattern, code, re.DOTALL))
        logger.debug(f"🧱 Found {len(class_matches)} class definitions")

        if file_type in ['.cpp', '.c', '.cc']:
            patterns = [
                # Statt des rekursiven Teils verwenden wir \{.*?\} (nicht gierig)
                ("class-scoped", r'(?:UFUNCTION\([^)]*\)\s*)?(?:virtual\s+)?(\w+(?:\s*<[^>]*>)?)\s+(\w+)::(\w+)\s*\((.*?)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?\s*\{.*?\}'),
                ("global",       r'(?:UFUNCTION\([^)]*\)\s*)?(?:virtual\s+)?(\w+(?:\s*<[^>]*>)?)\s+(\w+)\s*\((.*?)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?\s*\{.*?\}')
            ]

            for pattern_name, pattern in patterns:
                pattern_start = time.time()
                logger.debug(f"🔎 Matching {pattern_name} functions...")
                matches = list(re.finditer(pattern, code, re.DOTALL))
                logger.debug(f"✅ Found {len(matches)} {pattern_name} matches in {time.time() - pattern_start:.2f}s")

                for match in matches:
                    groups = match.groups()
                    if len(groups) == 4:  # Class-scoped function
                        return_type, class_name, func_name, params = groups
                    elif len(groups) == 3:  # Global function
                        return_type, func_name, params = groups
                        class_name = ""
                    else:
                        logger.warning("⚠️ Unexpected match group length, skipping.")
                        continue

                    full_text = match.group(0)
                    line_number = find_line_number(full_text, full_text.split('\n')[0], original_code)

                    functions.append({
                        "name": func_name,
                        "class_name": class_name,
                        "return_type": return_type,
                        "parameters": params,
                        "body": full_text.strip(),
                        "line_number": line_number
                    })

        else:
            logger.debug("🧠 Header mode: Extracting class-scoped functions...")
            for class_match in class_matches:
                class_name = class_match.group(1)
                class_body = class_match.group(2)
                class_start = class_match.start()

                patterns = [
                    ("header-func", r'(?:UFUNCTION\([^)]*\)\s*)?(?:virtual\s+)?(\w+(?:\s*<[^>]*>)?)\s+(\w+)\s*\((.*?)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:\{\s*(?:return[^;]*;)?\s*\})?;'),
                    ("property",    r'UPROPERTY\([^)]*\)\s*(\w+(?:\s*<[^>]*>)?)\s+(\w+)\s*\{[^}]*\};')
                ]

                for pattern_name, pattern in patterns:
                    logger.debug(f"🔎 Matching {pattern_name} in class {class_name}")
                    matches = list(re.finditer(pattern, class_body, re.DOTALL))
                    logger.debug(f"✅ Found {len(matches)} {pattern_name} matches")

                    for match in matches:
                        groups = match.groups()
                        if len(groups) >= 2:
                            return_type, func_name = groups[:2]
                            params = groups[2] if len(groups) > 2 else ""
                            full_text = match.group(0)
                            adjusted_text = code[class_start:class_start + match.start() + len(full_text)]
                            line_number = find_line_number(full_text, full_text.split('\n')[0], original_code)

                            functions.append({
                                "name": func_name,
                                "class_name": class_name,
                                "return_type": return_type,
                                "parameters": params,
                                "body": full_text.strip(),
                                "line_number": line_number
                            })

            logger.debug("🧠 Header mode: Global functions...")
            global_pattern = r'(?:UFUNCTION\([^)]*\)\s*)?(?:virtual\s+)?(\w+(?:\s*<[^>]*>)?)\s+(\w+)\s*\((.*?)\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?;'
            matches = list(re.finditer(global_pattern, code, re.DOTALL))
            logger.debug(f"✅ Found {len(matches)} global header functions")
            for match in matches:
                return_type, func_name, params = match.groups()
                full_text = match.group(0)
                line_number = find_line_number(full_text, full_text.split('\n')[0], original_code)

                functions.append({
                    "name": func_name,
                    "class_name": "",
                    "return_type": return_type,
                    "parameters": params,
                    "body": full_text.strip(),
                    "line_number": line_number
                })

    except Exception:
        logger.exception("🔥 Exception in extract_functions")

    logger.debug(f"✅ extract_functions DONE in {time.time() - start_time:.2f}s – {len(functions)} total functions found")
    return functions
    
def find_matching_functions(cpp_funcs: List[Dict[str, Any]], h_funcs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Finds matching function pairs between .cpp and .h files."""
    matched_pairs = []
    
    try:
        for cpp_func in cpp_funcs:
            for h_func in h_funcs:
                # Match based on function name, class name, and return type
                if (cpp_func["name"] == h_func["name"] and 
                    cpp_func["class_name"] == h_func["class_name"] and 
                    cpp_func["return_type"] == h_func["return_type"]):
                    
                    # Additional validation for parameters
                    cpp_params = re.sub(r'\s+', '', cpp_func["parameters"])
                    h_params = re.sub(r'\s+', '', h_func["parameters"])
                    
                    # Remove parameter names, keep only types
                    cpp_params = re.sub(r'(\w+)\s+(\w+)(?=[,)])', r'\1', cpp_params)
                    h_params = re.sub(r'(\w+)\s+(\w+)(?=[,)])', r'\1', h_params)
                    
                    if cpp_params == h_params:
                        matched_pairs.append((cpp_func, h_func))
                        logger.debug(f"Matched function: {cpp_func['name']} in class {cpp_func['class_name']}")
                        break
    except Exception as e:
        logger.error(f"🔥 Error matching functions: {str(e)}")
    
    return matched_pairs

def call_llm_api(code: str) -> str:
    """Calls the LLM API to generate the output."""
    if config.disable_api:
        return ""
        
    try:
        response = requests.post(config.api_endpoint, json={"code": code}, timeout=5)
        if response.status_code == 200:
            return response.json()["output"]
        else:
            logger.error(f"API call failed with status code: {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return ""

def get_output_paths(output_dir):
    """Get absolute paths for all output files."""
    output_dir = os.path.abspath(output_dir)
    return {
        'dir': output_dir,
        'combined': os.path.join(output_dir, "combined_dataset.jsonl"),
        'pretty': os.path.join(output_dir, "pretty_dataset.json")
    }

def ensure_dir_exists(path):
    """Ensure directory exists, create if it doesn't."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        logger.error(f"❌ Failed to create directory for {path}: {e}")
        return False
    return True

def normalize_path(path):
    """Normalize path to use system-specific separators."""
    return os.path.normpath(path)

def timestamped(msg):
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"

def prepare_dataset(file_paths):
    all_extracted_items = []
    file_contents = {}

    logger.info(timestamped("🚀 prepare_dataset started"))

    try:
        total_files = len(file_paths)
        logger.info(timestamped(f"📂 Total files to process: {total_files}"))

        # Step 1: Read files
        with tqdm(total=total_files, desc="Reading files") as pbar:
            for i, path in enumerate(file_paths, 1):
                logger.debug(timestamped(f"📄 [{i}/{total_files}] Reading file: {path}"))
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    if content.strip():
                        file_contents[path] = content
                        logger.debug(timestamped(f"✅ Read success: {path}"))
                    else:
                        logger.warning(timestamped(f"⚠️ Empty file skipped: {path}"))
                except Exception as e:
                    logger.exception(timestamped(f"🔥 Exception while reading file: {path}"))
                pbar.update(1)

        logger.debug(timestamped("📦 Grouping files..."))
        file_groups = {}
        for path in file_contents.keys():
            base_name = os.path.splitext(os.path.basename(path))[0]
            if base_name not in file_groups:
                file_groups[base_name] = {"cpp": None, "h": None}

            ext = os.path.splitext(path)[1].lower()
            if ext in ['.cpp', '.c', '.cc']:
                file_groups[base_name]["cpp"] = path
            elif ext == '.h':
                file_groups[base_name]["h"] = path

        total_groups = len(file_groups)
        logger.debug(timestamped(f"🧩 Grouped into {total_groups} file pairs"))

        # Step 2: Process file groups
        with tqdm(total=total_groups, desc="Processing file groups") as pbar:
            for i, (base_name, group) in enumerate(file_groups.items(), 1):
                logger.debug(timestamped(f"🔧 [{i}/{total_groups}] Processing group: {base_name}"))

                cpp_path = group["cpp"]
                h_path = group["h"]
                output_file = normalize_path(os.path.join(config.output_dir, f"{base_name}.jsonl"))

                if cpp_path and h_path:
                    logger.debug(timestamped(f"🧠 Paired file: {cpp_path} + {h_path}"))

                    if not ensure_dir_exists(output_file):
                        logger.error(timestamped(f"❌ Failed to ensure directory: {output_file}"))
                        continue

                    try:
                        cpp_content = file_contents[cpp_path]
                        h_content = file_contents[h_path]

                        cpp_training = clean_code_for_training(cpp_content)
                        h_training = clean_code_for_training(h_content)

                        cpp_funcs_training = extract_functions(cpp_training, '.cpp', cpp_content)
                        h_funcs_training = extract_functions(h_training, '.h', h_content)

                        logger.debug(timestamped(f"🔍 Found {len(cpp_funcs_training)} cpp + {len(h_funcs_training)} header funcs"))

                        matched_pairs = find_matching_functions(cpp_funcs_training, h_funcs_training)
                        logger.debug(timestamped(f"🔗 Matched {len(matched_pairs)} function pairs"))

                        if matched_pairs:
                            with open(output_file, "w", encoding="utf-8", newline="\n") as fout:
                                for pair_idx, (cpp_func, h_func) in enumerate(matched_pairs, 1):
                                    logger.debug(timestamped(f"✏️ Writing pair {pair_idx}/{len(matched_pairs)}"))
                                    class_scope = f" in class {cpp_func['class_name']}" if cpp_func['class_name'] else ""
                                    entry = {
                                        "instruction": f"Explain the following C++ implementation and its header{class_scope}.",
                                        "input": f"Header:\n{h_func['body'].strip()}\n\nImplementation:\n{cpp_func['body'].strip()}",
                                        "output": ""  # API is disabled
                                    }
                                    json.dump(entry, fout, ensure_ascii=False)
                                    fout.write("\n")

                                    training_item = {
                                        "type": "training",
                                        "language": "cpp",
                                        "header": h_func["body"].strip(),
                                        "implementation": cpp_func["body"].strip(),
                                        "file_path": normalize_path(cpp_path),
                                        "header_path": normalize_path(h_path),
                                        "function_name": cpp_func["name"],
                                        "class_name": cpp_func["class_name"],
                                        "return_type": cpp_func["return_type"],
                                        "line_number": cpp_func["line_number"],
                                        "header_line_number": h_func["line_number"],
                                        "training_file": output_file
                                    }
                                    all_extracted_items.append(training_item)
                            logger.debug(timestamped(f"💾 Saved training data to: {output_file}"))
                        else:
                            logger.warning(timestamped(f"⚠️ No function matches in group: {base_name}"))
                    except Exception as e:
                        logger.exception(timestamped(f"🔥 Exception during matched pair processing"))
                        continue

                # CPP-only
                if cpp_path:
                    try:
                        logger.debug(timestamped(f"🔍 Processing CPP file: {cpp_path}"))
                        cpp_content = file_contents[cpp_path]
                        cpp_combined = clean_code_for_combined(cpp_content)
                        cpp_funcs_combined = extract_functions(cpp_combined, '.cpp', cpp_content)

                        for func in cpp_funcs_combined:
                            item = {
                                "type": "combined",
                                "language": "cpp",
                                "code": func["body"],
                                "file_path": normalize_path(cpp_path),
                                "function_name": func["name"],
                                "class_name": func["class_name"],
                                "return_type": func["return_type"],
                                "line_number": func["line_number"],
                                "training_file": output_file
                            }
                            all_extracted_items.append(item)
                    except Exception:
                        logger.exception(timestamped(f"🔥 Error in CPP-only pass"))

                # Header-only
                if h_path:
                    try:
                        logger.debug(timestamped(f"🔍 Processing Header file: {h_path}"))
                        h_content = file_contents[h_path]
                        h_combined = clean_code_for_combined(h_content)
                        h_funcs_combined = extract_functions(h_combined, '.h', h_content)

                        for func in h_funcs_combined:
                            item = {
                                "type": "combined",
                                "language": "h",
                                "code": func["body"],
                                "file_path": normalize_path(h_path),
                                "function_name": func["name"],
                                "class_name": func["class_name"],
                                "return_type": func["return_type"],
                                "line_number": func["line_number"],
                                "training_file": output_file
                            }
                            all_extracted_items.append(item)
                    except Exception:
                        logger.exception(timestamped(f"🔥 Error in Header-only pass"))

                logger.debug(timestamped(f"✅ Finished group {i}/{total_groups}: {base_name}"))
                pbar.update(1)

    except Exception:
        logger.exception(timestamped(f"🔥 Unexpected top-level exception"))
    
    logger.info(timestamped(f"🏁 prepare_dataset finished with {len(all_extracted_items)} items"))
    return all_extracted_items

def save_file_datasets(files, all_extracted_items, output_dir):
    """Saves extracted data into separate files per source file."""
    items_by_file = {}
    for item in all_extracted_items:
        if item["type"] == "training":  # Only use training items
            file_path = item["file_path"]
            if file_path not in items_by_file:
                items_by_file[file_path] = []
            items_by_file[file_path].append(item)

    for file_path, items in items_by_file.items():
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.jsonl")
        
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8", newline="\n") as f:
                for item in items:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            logger.debug(f"💾 Saved training data to {output_file}")
        except Exception as e:
            logger.error(f"🔥 Error saving training data to {output_file}: {e}")

def save_combined_dataset(all_extracted_items, output_file):
    """Saves all extracted data into a single JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8", newline="\n") as f:
            for item in all_extracted_items:
                if item["type"] == "combined":  # Only use combined items
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
        logger.debug(f"💾 Saved combined data to {output_file}")
    except Exception as e:
        logger.error(f"🔥 Error saving combined data to {output_file}: {e}")

def save_pretty_dataset(all_extracted_items, output_file, output_dir):
    """Saves all extracted data into a single pretty-formatted JSON file."""
    pretty_data = []
    for item in all_extracted_items:
        if item["type"] == "combined":  # Only use combined items
            class_scope = f" in class {item['class_name']}" if item['class_name'] else ""
            
            pretty_data.append({
                "instruction": f"Explain the following {item['language']}-Code{class_scope}.",
                "input": item["code"].strip(),
                "output": call_llm_api(item["code"].strip()),
                "file": item["file_path"],
                "line": item["line_number"],
                "training_file": item["training_file"]
            })

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8", newline="\n") as fpretty:
            json.dump(pretty_data, fpretty, indent=4, ensure_ascii=False)
        logger.debug(f"💾 Saved pretty data to {output_file}")
    except Exception as e:
        logger.error(f"🔥 Error saving pretty data to {output_file}: {e}")

def read_file_content(path):
    """Tries to read the file content with different encodings."""
    encodings = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception:
            continue
    logger.error(f"🔥 Error reading file: {path}")
    return ""

def main():
    parser = argparse.ArgumentParser(description='Process C++ files for fine-tuning data preparation.')
    parser.add_argument('--disable-api', action='store_true', help='Disable LLM API calls')
    parser.add_argument('--input-dirs', nargs='*', help='Input directories or files to process (optional)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of files to process in each batch')
    parser.add_argument('--output-dir', type=str, default=config.output_dir, help='Output directory for generated files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Update configuration
    config.disable_api = args.disable_api
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.debug = args.debug

    # Configure logging based on debug flag
    log_level = logging.DEBUG if config.debug else logging.INFO
    logger.setLevel(log_level)
    
    if config.disable_api:
        logger.info("API calls are disabled")

    # Get output paths
    output_paths = get_output_paths(config.output_dir)
    output_dir = output_paths['dir']
    combined_output = output_paths['combined']
    pretty_output = output_paths['pretty']

    directories = []
    
    # If paths were provided as arguments, use those
    if args.input_dirs:
        directories.extend(args.input_dirs)
    else:
        # Otherwise ask for paths interactively
        print("Enter the directory or file paths, one per line. Press Enter with empty line to finish.")
        while True:
            inp = input("Path (or press Enter to finish): ").strip()
            if not inp:
                break
            # Convert to absolute path
            abs_path = os.path.abspath(inp)
            if os.path.exists(abs_path):
                directories.append(abs_path)
            else:
                logger.error(f"❌ Invalid path: {inp}")

    if not directories:
        logger.error("❌ No paths provided. Exiting.")
        return

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to create output directory {output_dir}: {e}")
        return

    # Collect and process files
    all_files = []
    for path in tqdm(directories, desc="Collecting files", unit="dir"):
        dir_files = collect_file_paths(path)
        all_files.extend(dir_files)

    if not all_files:
        logger.error("❌ No files found to process. Check your input paths and file extensions.")
        return

    logger.info(f"\n🔍 Total files to process: {len(all_files)}")

    try:
        all_extracted_items = prepare_dataset(all_files)
        
        # Save the datasets with progress bars
        with tqdm(total=3, desc="Saving datasets") as pbar:
            save_file_datasets(all_files, all_extracted_items, output_dir)
            pbar.update(1)
            save_combined_dataset(all_extracted_items, combined_output)
            pbar.update(1)
            save_pretty_dataset(all_extracted_items, pretty_output, output_dir)
            pbar.update(1)
        
        # Print summary
        logger.info("\n✅ Processing complete!")
        logger.info(f"📊 Summary:")
        logger.info(f"  - Processed {len(all_files)} files")
        logger.info(f"  - Extracted {len(all_extracted_items)} code items")
        logger.info(f"  - Output directory: {output_dir}")
        logger.info(f"  - Combined dataset: {combined_output}")
        logger.info(f"  - Pretty dataset: {pretty_output}")
        
    except Exception as e:
        logger.error(f"🔥 Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        with logging_redirect_tqdm():
            main()
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        logger.error(f"🔥 Unexpected error: {e}")
        raise