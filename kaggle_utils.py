"""
Kaggle API utilities for AgentLaboratory.

Provides functions for interacting with the Kaggle API:
- Listing competitions and datasets
- Submitting notebooks for execution
- Configuring compute (CPU vs GPU)
- Real-time log monitoring for training runs
- Status polling with early failure detection

Requires KAGGLE_API_TOKEN environment variable to be set.
"""

import os
import json
import time
import logging
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)


def get_kaggle_api():
    """
    Initialize and return the Kaggle API client.
    Requires KAGGLE_API_TOKEN environment variable.

    @return: (KaggleApi) authenticated Kaggle API client
    @raises EnvironmentError: if KAGGLE_API_TOKEN is not set
    """
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "KAGGLE_API_TOKEN environment variable is not set. "
            "Get your token from https://www.kaggle.com/settings -> API -> Create New Token, "
            "then set: export KAGGLE_API_TOKEN=<your-token>"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "kaggle package is not installed. Install it with: pip install kaggle"
        )

    api = KaggleApi()
    api.authenticate()
    return api


def list_competitions(search_term=None, page=1):
    """
    List Kaggle competitions.

    @param search_term: (str, optional) search filter for competitions
    @param page: (int) page number for pagination
    @return: (list) list of competition objects
    """
    api = get_kaggle_api()
    competitions = api.competitions_list(search=search_term or "", page=page)
    return competitions


def list_datasets(search_term=None, page=1):
    """
    List Kaggle datasets.

    @param search_term: (str, optional) search filter for datasets
    @param page: (int) page number for pagination
    @return: (list) list of dataset objects
    """
    api = get_kaggle_api()
    datasets = api.dataset_list(search=search_term or "", page=page)
    return datasets


def download_dataset(dataset_ref, path="./data"):
    """
    Download a Kaggle dataset.

    @param dataset_ref: (str) dataset reference in format 'owner/dataset-name'
    @param path: (str) local path to download to
    """
    api = get_kaggle_api()
    api.dataset_download_files(dataset_ref, path=path, unzip=True)


def create_notebook_metadata(title, code_file, language="python", kernel_type="script",
                             enable_gpu=False, enable_internet=True, dataset_sources=None,
                             competition_sources=None):
    """
    Create a Kaggle kernel-metadata.json for notebook push.

    @param title: (str) notebook title
    @param code_file: (str) path to the code file
    @param language: (str) programming language ('python' or 'r')
    @param kernel_type: (str) 'script' or 'notebook'
    @param enable_gpu: (bool) whether to enable GPU
    @param enable_internet: (bool) whether to enable internet access
    @param dataset_sources: (list, optional) list of dataset references
    @param competition_sources: (list, optional) list of competition references
    @return: (dict) metadata dictionary
    """
    api = get_kaggle_api()
    username = api.get_config_value(api.CONFIG_NAME_USER)

    metadata = {
        "id": f"{username}/{title.lower().replace(' ', '-')}",
        "title": title,
        "code_file": code_file,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": competition_sources or [],
    }
    return metadata


def push_notebook(folder_path):
    """
    Push a notebook to Kaggle for execution.

    @param folder_path: (str) path to folder containing kernel-metadata.json and code file
    """
    api = get_kaggle_api()
    api.kernels_push(folder_path)


def get_notebook_status(kernel_ref):
    """
    Get the status of a Kaggle notebook execution.

    @param kernel_ref: (str) kernel reference in format 'username/kernel-name'
    @return: (dict) status information
    """
    api = get_kaggle_api()
    return api.kernels_status(kernel_ref)


def get_notebook_output(kernel_ref, path="./output"):
    """
    Download output from a completed Kaggle notebook.

    @param kernel_ref: (str) kernel reference in format 'username/kernel-name'
    @param path: (str) local path to download output to
    """
    api = get_kaggle_api()
    api.kernels_output(kernel_ref, path=path)


def get_compute_mode():
    """
    Get the current compute mode from environment.

    @return: (str) 'cpu' or 'gpu'
    """
    return os.getenv("COMPUTE_MODE", "cpu").lower()


def is_gpu_available():
    """
    Check if GPU is available in the current environment.

    @return: (bool) True if GPU is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def configure_device():
    """
    Configure the compute device based on COMPUTE_MODE environment variable
    and hardware availability.

    @return: (str) device string for PyTorch ('cuda' or 'cpu')
    """
    mode = get_compute_mode()
    if mode == "gpu" and is_gpu_available():
        return "cuda"
    return "cpu"


def submit_training_notebook(title, code_file, enable_gpu=True, dataset_sources=None,
                             competition_sources=None, output_path="./kaggle_output"):
    """
    Submit a training notebook to Kaggle for remote execution.

    Creates metadata, pushes the notebook, and returns the kernel reference
    for status tracking.

    @param title: (str) notebook title
    @param code_file: (str) path to the Python code file
    @param enable_gpu: (bool) whether to enable GPU on Kaggle
    @param dataset_sources: (list, optional) dataset references to attach
    @param competition_sources: (list, optional) competition references to attach
    @param output_path: (str) path for output directory
    @return: (str) kernel reference for tracking
    """
    folder = tempfile.mkdtemp()
    try:
        metadata = create_notebook_metadata(
            title=title,
            code_file=os.path.basename(code_file),
            enable_gpu=enable_gpu,
            dataset_sources=dataset_sources,
            competition_sources=competition_sources,
        )

        with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        shutil.copy2(code_file, os.path.join(folder, os.path.basename(code_file)))

        push_notebook(folder)

        return metadata["id"]
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def poll_notebook_status(kernel_ref, poll_interval=15, timeout=1800):
    """
    Poll a Kaggle notebook's status until completion, failure, or timeout.
    Returns status updates as they occur for real-time monitoring.

    @param kernel_ref: (str) kernel reference 'username/kernel-name'
    @param poll_interval: (int) seconds between status checks
    @param timeout: (int) maximum seconds to wait
    @return: (dict) final status with 'status' and 'failure_message' keys
    """
    start_time = time.time()
    last_status = None

    while (time.time() - start_time) < timeout:
        try:
            status = get_notebook_status(kernel_ref)
            status_str = str(status) if status else "unknown"

            if status_str != last_status:
                elapsed = int(time.time() - start_time)
                logger.info(f"[{elapsed}s] Kernel {kernel_ref}: {status_str}")
                last_status = status_str

            if "complete" in status_str.lower():
                return {"status": "complete", "failure_message": None,
                        "elapsed": int(time.time() - start_time)}
            if "error" in status_str.lower() or "cancel" in status_str.lower():
                return {"status": "error", "failure_message": status_str,
                        "elapsed": int(time.time() - start_time)}
        except Exception as e:
            logger.warning(f"Status poll error for {kernel_ref}: {e}")

        time.sleep(poll_interval)

    return {"status": "timeout", "failure_message": f"Timed out after {timeout}s",
            "elapsed": timeout}


def retrieve_notebook_logs(kernel_ref, output_dir=None):
    """
    Retrieve and parse logs from a completed Kaggle notebook execution.
    Downloads output and extracts log content for analysis.

    @param kernel_ref: (str) kernel reference 'username/kernel-name'
    @param output_dir: (str, optional) directory to save output
    @return: (dict) with 'log_content', 'output_files', 'errors' keys
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="kaggle_logs_")

    result = {"log_content": "", "output_files": [], "errors": []}

    try:
        get_notebook_output(kernel_ref, path=output_dir)

        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            result["output_files"].append(fpath)
            if fname.endswith(".log") or fname.endswith(".txt"):
                with open(fpath, "r", errors="replace") as f:
                    content = f.read()
                result["log_content"] += f"\n=== {fname} ===\n{content}"
                for line in content.split("\n"):
                    lower = line.lower()
                    if any(kw in lower for kw in ["error", "exception",
                                                   "traceback", "failed"]):
                        result["errors"].append(line.strip())
    except Exception as e:
        result["errors"].append(f"Failed to retrieve logs: {e}")
        logger.error(f"Log retrieval failed for {kernel_ref}: {e}")

    return result


def inject_logging_code(code):
    """
    Inject logging and monitoring code into a training script.
    Adds structured print statements for real-time progress tracking
    that can be parsed from Kaggle notebook output.

    @param code: (str) original Python training code
    @return: (str) code with logging injected
    """
    logging_header = '''
import sys
import time
import traceback

_PIPELINE_START = time.time()

def pipeline_log(msg, level="INFO"):
    elapsed = time.time() - _PIPELINE_START
    print(f"[PIPELINE {level} {elapsed:.1f}s] {msg}", flush=True)

pipeline_log("Training script started")
pipeline_log(f"Python version: {sys.version}")

try:
    import torch
    pipeline_log(f"PyTorch version: {torch.__version__}")
    pipeline_log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        pipeline_log(f"GPU: {torch.cuda.get_device_name(0)}")
        pipeline_log(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
except ImportError:
    pipeline_log("PyTorch not available", level="WARN")

'''

    logging_footer = '''

pipeline_log("Training script completed successfully")
pipeline_log(f"Total time: {time.time() - _PIPELINE_START:.1f}s")
'''

    error_wrapper_start = '''
try:
'''
    error_wrapper_end = '''
except Exception as _pipeline_err:
    pipeline_log(f"FATAL ERROR: {type(_pipeline_err).__name__}: {_pipeline_err}", level="ERROR")
    pipeline_log(traceback.format_exc(), level="ERROR")
    raise
'''

    indented_code = "\n".join("    " + line for line in code.split("\n"))
    return (logging_header + error_wrapper_start + indented_code +
            error_wrapper_end + logging_footer)


def parse_pipeline_logs(log_text):
    """
    Parse structured pipeline logs from notebook output.

    @param log_text: (str) raw log text from notebook output
    @return: (dict) parsed log with 'messages', 'errors', 'warnings', 'metrics'
    """
    result = {"messages": [], "errors": [], "warnings": [], "metrics": {}}

    for line in log_text.split("\n"):
        line = line.strip()
        if not line.startswith("[PIPELINE"):
            continue

        try:
            parts = line.split("]", 1)
            header = parts[0].replace("[PIPELINE ", "")
            msg = parts[1].strip() if len(parts) > 1 else ""

            tokens = header.split()
            level = tokens[0] if tokens else "INFO"
            timestamp = tokens[1] if len(tokens) > 1 else "0s"

            entry = {"level": level, "time": timestamp, "message": msg}
            result["messages"].append(entry)

            if level == "ERROR":
                result["errors"].append(msg)
            elif level == "WARN":
                result["warnings"].append(msg)

            if "loss:" in msg.lower() or "accuracy:" in msg.lower():
                result["metrics"][timestamp] = msg
        except (IndexError, ValueError):
            continue

    return result
