"""
Kaggle API utilities for AgentLaboratory.

Provides functions for interacting with the Kaggle API:
- Listing competitions and datasets
- Submitting notebooks for execution
- Configuring compute (CPU vs GPU)

Requires KAGGLE_API_TOKEN environment variable to be set.
"""

import os
import json
import subprocess


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
    import tempfile
    import shutil

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
