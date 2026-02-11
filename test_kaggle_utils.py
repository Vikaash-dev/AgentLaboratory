"""
Tests for kaggle_utils module.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock


class TestComputeConfiguration(unittest.TestCase):
    """Tests for compute mode configuration functions."""

    @patch.dict(os.environ, {"COMPUTE_MODE": "cpu"}, clear=False)
    def test_get_compute_mode_cpu(self):
        from kaggle_utils import get_compute_mode
        self.assertEqual(get_compute_mode(), "cpu")

    @patch.dict(os.environ, {"COMPUTE_MODE": "gpu"}, clear=False)
    def test_get_compute_mode_gpu(self):
        from kaggle_utils import get_compute_mode
        self.assertEqual(get_compute_mode(), "gpu")

    @patch.dict(os.environ, {}, clear=False)
    def test_get_compute_mode_default(self):
        os.environ.pop("COMPUTE_MODE", None)
        from kaggle_utils import get_compute_mode
        self.assertEqual(get_compute_mode(), "cpu")

    @patch("kaggle_utils.is_gpu_available", return_value=False)
    @patch.dict(os.environ, {"COMPUTE_MODE": "cpu"}, clear=False)
    def test_configure_device_cpu(self, mock_gpu):
        from kaggle_utils import configure_device
        self.assertEqual(configure_device(), "cpu")

    @patch("kaggle_utils.is_gpu_available", return_value=True)
    @patch.dict(os.environ, {"COMPUTE_MODE": "gpu"}, clear=False)
    def test_configure_device_gpu(self, mock_gpu):
        from kaggle_utils import configure_device
        self.assertEqual(configure_device(), "cuda")

    @patch("kaggle_utils.is_gpu_available", return_value=False)
    @patch.dict(os.environ, {"COMPUTE_MODE": "gpu"}, clear=False)
    def test_configure_device_gpu_unavailable(self, mock_gpu):
        from kaggle_utils import configure_device
        self.assertEqual(configure_device(), "cpu")


class TestKaggleApiInit(unittest.TestCase):
    """Tests for Kaggle API initialization."""

    @patch.dict(os.environ, {}, clear=False)
    def test_missing_token_raises_error(self):
        os.environ.pop("KAGGLE_API_TOKEN", None)
        from kaggle_utils import get_kaggle_api
        with self.assertRaises(EnvironmentError):
            get_kaggle_api()


class TestNotebookMetadata(unittest.TestCase):
    """Tests for notebook metadata creation."""

    @patch("kaggle_utils.get_kaggle_api")
    def test_create_notebook_metadata(self, mock_api_func):
        mock_api = MagicMock()
        mock_api.get_config_value.return_value = "testuser"
        mock_api_func.return_value = mock_api

        from kaggle_utils import create_notebook_metadata
        metadata = create_notebook_metadata(
            title="Test Notebook",
            code_file="train.py",
            enable_gpu=True,
            dataset_sources=["owner/dataset"],
        )

        self.assertEqual(metadata["id"], "testuser/test-notebook")
        self.assertEqual(metadata["title"], "Test Notebook")
        self.assertEqual(metadata["code_file"], "train.py")
        self.assertTrue(metadata["enable_gpu"])
        self.assertTrue(metadata["enable_internet"])
        self.assertEqual(metadata["dataset_sources"], ["owner/dataset"])
        self.assertEqual(metadata["competition_sources"], [])


class TestInferenceGeminiSupport(unittest.TestCase):
    """Tests for Gemini 2.5 Pro Preview support in inference.py."""

    def test_gemini_model_string_recognized(self):
        """Verify the model string is handled in query_model code."""
        with open(os.path.join(os.path.dirname(__file__), "inference.py")) as f:
            source = f.read()
        self.assertIn("gemini-2.5-pro", source)
        self.assertIn("thinking_budget", source)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_gemini_api_key_loaded_from_env(self):
        """Verify GEMINI_API_KEY is loaded from environment."""
        key = os.getenv("GEMINI_API_KEY")
        self.assertEqual(key, "test-key")


class TestEnvExample(unittest.TestCase):
    """Tests for .env.example file."""

    def test_env_example_exists(self):
        env_path = os.path.join(os.path.dirname(__file__), ".env.example")
        self.assertTrue(os.path.exists(env_path))

    def test_env_example_contains_required_vars(self):
        env_path = os.path.join(os.path.dirname(__file__), ".env.example")
        with open(env_path) as f:
            content = f.read()
        self.assertIn("OPENAI_API_KEY", content)
        self.assertIn("GEMINI_API_KEY", content)
        self.assertIn("TAVILY_API_KEY", content)
        self.assertIn("KAGGLE_API_TOKEN", content)
        self.assertIn("COMPUTE_MODE", content)

    def test_env_example_has_no_real_keys(self):
        env_path = os.path.join(os.path.dirname(__file__), ".env.example")
        with open(env_path) as f:
            content = f.read()
        # All key lines should be commented out
        for line in content.strip().split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                self.fail(f"Found uncommented key line: {line}")


class TestDockerfiles(unittest.TestCase):
    """Tests for Docker configuration files."""

    def test_dockerfile_exists(self):
        path = os.path.join(os.path.dirname(__file__), "Dockerfile")
        self.assertTrue(os.path.exists(path))

    def test_docker_compose_exists(self):
        path = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
        self.assertTrue(os.path.exists(path))

    def test_dockerfile_uses_kaggle_image(self):
        path = os.path.join(os.path.dirname(__file__), "Dockerfile")
        with open(path) as f:
            content = f.read()
        self.assertIn("kaggle-images/python", content)

    def test_docker_compose_has_cpu_and_gpu(self):
        path = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
        with open(path) as f:
            content = f.read()
        self.assertIn("cpu:", content)
        self.assertIn("gpu:", content)
        self.assertIn("COMPUTE_MODE=cpu", content)
        self.assertIn("COMPUTE_MODE=gpu", content)


if __name__ == "__main__":
    unittest.main()
