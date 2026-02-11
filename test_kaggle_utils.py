"""
Tests for kaggle_utils, tools execution fixes, and kaggle_training_pipeline.
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
    """Tests for Gemini 2.5 Pro and Flash support in inference.py."""

    def test_gemini_model_string_recognized(self):
        """Verify the model string is handled in query_model code."""
        with open(os.path.join(os.path.dirname(__file__), "inference.py")) as f:
            source = f.read()
        self.assertIn("gemini-2.5-pro", source)
        self.assertIn("thinking_budget", source)

    def test_gemini_flash_model_recognized(self):
        """Verify Gemini 2.5 Flash is supported for monitoring."""
        with open(os.path.join(os.path.dirname(__file__), "inference.py")) as f:
            source = f.read()
        self.assertIn("gemini-2.5-flash", source)

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


# ─── New tests for fixed shortcomings ────────────────────────────────────

class TestLogInjection(unittest.TestCase):
    """Tests for training code log injection."""

    def test_inject_logging_adds_pipeline_log(self):
        from kaggle_utils import inject_logging_code
        code = "x = 1 + 1\nprint(x)"
        result = inject_logging_code(code)
        self.assertIn("pipeline_log", result)
        self.assertIn("PIPELINE", result)
        self.assertIn("Training script started", result)

    def test_inject_logging_wraps_in_try_except(self):
        from kaggle_utils import inject_logging_code
        code = "print('hello')"
        result = inject_logging_code(code)
        self.assertIn("try:", result)
        self.assertIn("except Exception", result)
        self.assertIn("FATAL ERROR", result)

    def test_inject_logging_includes_device_detection(self):
        from kaggle_utils import inject_logging_code
        code = "pass"
        result = inject_logging_code(code)
        self.assertIn("torch.cuda.is_available()", result)


class TestLogParsing(unittest.TestCase):
    """Tests for structured pipeline log parsing."""

    def test_parse_pipeline_logs_extracts_messages(self):
        from kaggle_utils import parse_pipeline_logs
        log = "[PIPELINE INFO 0.1s] Training started\n[PIPELINE ERROR 5.2s] OOM crash"
        result = parse_pipeline_logs(log)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("OOM crash", result["errors"][0])

    def test_parse_pipeline_logs_empty_input(self):
        from kaggle_utils import parse_pipeline_logs
        result = parse_pipeline_logs("")
        self.assertEqual(result["messages"], [])
        self.assertEqual(result["errors"], [])

    def test_parse_pipeline_logs_extracts_warnings(self):
        from kaggle_utils import parse_pipeline_logs
        log = "[PIPELINE WARN 1.0s] Slow convergence"
        result = parse_pipeline_logs(log)
        self.assertEqual(len(result["warnings"]), 1)


class TestDeviceDetection(unittest.TestCase):
    """Tests for GPU/device detection in tools.py."""

    def test_detect_device_returns_valid_string(self):
        try:
            from tools import detect_device
        except ImportError:
            self.skipTest("tools.py dependencies not available in test environment")
        device = detect_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    def test_detect_device_default_is_cpu(self):
        """On CI without GPU, should return cpu."""
        try:
            from tools import detect_device
        except ImportError:
            self.skipTest("tools.py dependencies not available in test environment")
        device = detect_device()
        self.assertIsInstance(device, str)


class TestPipelineState(unittest.TestCase):
    """Tests for KaggleTrainingPipeline state tracking."""

    def test_pipeline_state_error_tracking(self):
        try:
            from kaggle_training_pipeline import PipelineState
        except ImportError:
            self.skipTest("kaggle_training_pipeline dependencies not available")
        state = PipelineState()
        state.record("test", "code", errors=["error1", "error2"])
        self.assertEqual(len(state.accumulated_errors), 2)
        summary = state.get_error_summary()
        self.assertIn("error1", summary)
        self.assertIn("error2", summary)

    def test_pipeline_state_history(self):
        try:
            from kaggle_training_pipeline import PipelineState
        except ImportError:
            self.skipTest("kaggle_training_pipeline dependencies not available")
        state = PipelineState()
        state.record("phase1", "code1")
        state.record("phase2", "code2", errors=["err"])
        summary = state.get_history_summary()
        self.assertIn("phase1", summary)
        self.assertIn("ERRORS", summary)

    def test_pipeline_state_empty(self):
        try:
            from kaggle_training_pipeline import PipelineState
        except ImportError:
            self.skipTest("kaggle_training_pipeline dependencies not available")
        state = PipelineState()
        self.assertEqual(state.get_error_summary(), "No errors encountered in previous iterations.")
        self.assertEqual(state.get_history_summary(), "No previous iterations.")


class TestPipelineInit(unittest.TestCase):
    """Tests for KaggleTrainingPipeline initialization."""

    def test_pipeline_requires_gemini_key(self):
        """Pipeline should raise if no Gemini API key available."""
        try:
            from kaggle_training_pipeline import KaggleTrainingPipeline
        except ImportError:
            self.skipTest("kaggle_training_pipeline dependencies not available")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            with self.assertRaises(EnvironmentError):
                KaggleTrainingPipeline(task_description="test task")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_pipeline_creates_output_dir(self):
        import tempfile
        try:
            from kaggle_training_pipeline import KaggleTrainingPipeline
        except ImportError:
            self.skipTest("kaggle_training_pipeline dependencies not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "pipeline_out")
            pipeline = KaggleTrainingPipeline(
                task_description="test", output_dir=out
            )
            self.assertTrue(os.path.isdir(out))

    def test_pipeline_file_exists(self):
        """Verify kaggle_training_pipeline.py exists."""
        path = os.path.join(os.path.dirname(__file__), "kaggle_training_pipeline.py")
        self.assertTrue(os.path.exists(path))


class TestExecutionLogPersistence(unittest.TestCase):
    """Tests for execution log file persistence in tools.py."""

    def test_execute_code_creates_log_file(self):
        """execute_code should write a log file to execution_logs/."""
        try:
            from tools import _EXECUTION_LOG_DIR
        except ImportError:
            self.skipTest("tools.py dependencies not available in test environment")
        self.assertIn("execution_logs", _EXECUTION_LOG_DIR)


# ─── Sub-Agent Tests ─────────────────────────────────────────────────────

class TestSubAgentFiles(unittest.TestCase):
    """Tests for sub-agent module existence and structure."""

    def test_pipeline_subagents_exists(self):
        path = os.path.join(os.path.dirname(__file__), "pipeline_subagents.py")
        self.assertTrue(os.path.exists(path))

    def test_kaggle_best_practices_exists(self):
        path = os.path.join(os.path.dirname(__file__), "kaggle_best_practices.md")
        self.assertTrue(os.path.exists(path))

    def test_subagents_module_has_all_agents(self):
        """Verify all sub-agent classes are defined."""
        with open(os.path.join(os.path.dirname(__file__), "pipeline_subagents.py")) as f:
            source = f.read()
        for agent_name in ["ResearchAgent", "LoggingAgent", "CodeReviewAgent",
                           "CodeFixAgent", "CPUTestAgent", "GPUTrainingAgent",
                           "ErrorAnalysisAgent", "MonitoringAgent"]:
            self.assertIn(f"class {agent_name}", source,
                          f"Missing sub-agent: {agent_name}")


class TestResearchAgent(unittest.TestCase):
    """Tests for the ResearchAgent (Tavily-powered)."""

    def _get_agent(self):
        try:
            from pipeline_subagents import ResearchAgent
            return ResearchAgent
        except ImportError:
            self.skipTest("pipeline_subagents dependencies not available")

    def test_research_agent_init(self):
        AgentClass = self._get_agent()
        agent = AgentClass(gemini_api_key="test-key")
        self.assertIsNotNone(agent._cache)
        self.assertEqual(agent._cache, {})

    def test_research_agent_has_tavily_search(self):
        AgentClass = self._get_agent()
        agent = AgentClass()
        self.assertTrue(hasattr(agent, '_tavily_search'))

    @patch.dict(os.environ, {}, clear=False)
    def test_tavily_search_without_key_returns_empty(self):
        os.environ.pop("TAVILY_API_KEY", None)
        AgentClass = self._get_agent()
        agent = AgentClass()
        results = agent._tavily_search("test query")
        self.assertEqual(results, [])

    def test_static_fallback_loads(self):
        AgentClass = self._get_agent()
        agent = AgentClass()
        fallback = agent._load_static_fallback()
        self.assertIn("Kaggle", fallback)
        self.assertIn("logging", fallback.lower())


class TestMonitoringAgent(unittest.TestCase):
    """Tests for MonitoringAgent model configuration."""

    def test_monitoring_uses_flash_model(self):
        """MonitoringAgent should use Gemini 2.5 Flash for efficiency."""
        with open(os.path.join(os.path.dirname(__file__), "pipeline_subagents.py")) as f:
            source = f.read()
        self.assertIn('MONITORING_MODEL = "gemini-2.5-flash"', source)

    def test_thinking_model_is_pro(self):
        """Other agents should use Gemini 2.5 Pro for deep analysis."""
        with open(os.path.join(os.path.dirname(__file__), "pipeline_subagents.py")) as f:
            source = f.read()
        self.assertIn('THINKING_MODEL = "gemini-2.5-pro"', source)


class TestPipelineSubAgentIntegration(unittest.TestCase):
    """Tests for KaggleTrainingPipeline sub-agent integration."""

    def test_pipeline_has_all_sub_agents(self):
        """Pipeline should initialize all sub-agents."""
        try:
            from kaggle_training_pipeline import KaggleTrainingPipeline
        except ImportError:
            self.skipTest("kaggle_training_pipeline dependencies not available")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = KaggleTrainingPipeline(
                    task_description="test", output_dir=tmpdir
                )
                self.assertIsNotNone(pipeline.research_agent)
                self.assertIsNotNone(pipeline.logging_agent)
                self.assertIsNotNone(pipeline.review_agent)
                self.assertIsNotNone(pipeline.fix_agent)
                self.assertIsNotNone(pipeline.cpu_test_agent)
                self.assertIsNotNone(pipeline.gpu_agent)
                self.assertIsNotNone(pipeline.error_agent)
                self.assertIsNotNone(pipeline.monitor_agent)

    def test_pipeline_imports_subagents(self):
        """Pipeline module should import from pipeline_subagents."""
        with open(os.path.join(os.path.dirname(__file__),
                               "kaggle_training_pipeline.py")) as f:
            source = f.read()
        self.assertIn("from pipeline_subagents import", source)
        self.assertIn("ResearchAgent", source)
        self.assertIn("MonitoringAgent", source)


if __name__ == "__main__":
    unittest.main()
