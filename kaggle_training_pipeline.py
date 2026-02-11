"""
Kaggle Training Pipeline — Self-Iterating ML Training with Sub-Agents.

This pipeline uses specialized sub-agents for each phase of ML training:

    ResearchAgent (Tavily) → gathers best practices, docs, sample code
    LoggingAgent           → injects structured logging into training code
    CodeReviewAgent        → self-reviews code for errors and Kaggle constraints
    CodeFixAgent           → fixes issues found by review or error analysis
    CPUTestAgent           → creates CPU-only version for quick validation
    GPUTrainingAgent       → creates full GPU-accelerated training code
    ErrorAnalysisAgent     → analyzes logs/errors and diagnoses problems
    MonitoringAgent (Flash)→ lightweight real-time training progress monitoring

Workflow per iteration:
    1. ResearchAgent gathers context (Tavily search + static best practices)
    2. Generate training code → LoggingAgent adds monitoring
    3. CodeReviewAgent reviews → CodeFixAgent fixes issues
    4. CPUTestAgent creates CPU version → submit to Kaggle CPU → monitor
    5. ErrorAnalysisAgent checks CPU logs → CodeFixAgent fixes if needed
    6. GPUTrainingAgent creates GPU version → CodeReviewAgent reviews
    7. Submit to Kaggle GPU → MonitoringAgent watches progress
    8. Iterate with learned context

Usage:
    from kaggle_training_pipeline import KaggleTrainingPipeline
    pipeline = KaggleTrainingPipeline(task_description="Train a text classifier")
    pipeline.run(max_iterations=3)
"""

import os
import json
import time
import logging
import tempfile
import threading
from datetime import datetime

from inference import query_model
from kaggle_utils import (
    inject_logging_code,
    parse_pipeline_logs,
    submit_training_notebook,
    poll_notebook_status,
    retrieve_notebook_logs,
    get_compute_mode,
    configure_device,
)
from pipeline_subagents import (
    ResearchAgent,
    LoggingAgent,
    CodeReviewAgent,
    CodeFixAgent,
    CPUTestAgent,
    GPUTrainingAgent,
    ErrorAnalysisAgent,
    MonitoringAgent,
    THINKING_MODEL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Gemini model used for code generation
PIPELINE_MODEL = THINKING_MODEL


class PipelineState:
    """Tracks state across iterations for the self-improving pipeline."""

    def __init__(self):
        self.iteration = 0
        self.history = []       # list of {iteration, phase, code, logs, errors, fixes}
        self.best_code = None
        self.best_metrics = None
        self.accumulated_errors = []

    def record(self, phase, code, logs="", errors=None, metrics=None):
        entry = {
            "iteration": self.iteration,
            "phase": phase,
            "code_snippet": code[:500] if code else "",
            "logs_snippet": logs[:1000] if logs else "",
            "errors": errors or [],
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)
        if errors:
            self.accumulated_errors.extend(errors)

    def get_error_summary(self):
        """Summarize all past errors for the LLM to learn from."""
        if not self.accumulated_errors:
            return "No errors encountered in previous iterations."
        unique = list(dict.fromkeys(self.accumulated_errors))  # dedupe, preserve order
        return "\n".join(f"- {e}" for e in unique[-10:])  # last 10 unique errors

    def get_history_summary(self):
        """Get a compact summary of all iterations for context."""
        if not self.history:
            return "No previous iterations."
        lines = []
        for h in self.history[-6:]:  # last 6 entries
            status = "ERRORS" if h["errors"] else "OK"
            lines.append(f"  Iter {h['iteration']} / {h['phase']}: {status}")
            if h["errors"]:
                lines.append(f"    Errors: {'; '.join(h['errors'][:2])}")
        return "\n".join(lines)


class KaggleTrainingPipeline:
    """
    Self-iterating ML training pipeline using Kaggle notebooks and
    specialized sub-agents.

    Sub-agents:
        research_agent   — Tavily web search for best practices and docs
        logging_agent    — Injects structured logging into code
        review_agent     — Reviews code for errors and Kaggle constraints
        fix_agent        — Fixes issues from review or error analysis
        cpu_test_agent   — Creates CPU-only validation version
        gpu_agent        — Creates full GPU-accelerated training version
        error_agent      — Analyzes execution logs and diagnoses problems
        monitor_agent    — Gemini 3 Flash for real-time progress monitoring

    Workflow per iteration:
        1. ResearchAgent gathers context (Tavily + static best practices)
        2. Generate code → LoggingAgent adds monitoring → CodeReviewAgent reviews
        3. CPUTestAgent creates CPU version → submit to Kaggle CPU
        4. MonitoringAgent watches logs → ErrorAnalysisAgent diagnoses failures
        5. GPUTrainingAgent creates GPU version → submit to Kaggle GPU
        6. MonitoringAgent watches GPU training → iterate with learned context
    """

    def __init__(self, task_description, dataset_sources=None,
                 competition_sources=None, gemini_api_key=None,
                 output_dir="./kaggle_pipeline_output"):
        """
        @param task_description: (str) what model to train, e.g. "Train MNIST classifier"
        @param dataset_sources: (list) Kaggle dataset refs to attach
        @param competition_sources: (list) Kaggle competition refs to attach
        @param gemini_api_key: (str, optional) Gemini API key, reads from env if None
        @param output_dir: (str) directory for pipeline outputs
        """
        self.task = task_description
        self.dataset_sources = dataset_sources or []
        self.competition_sources = competition_sources or []
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.output_dir = output_dir
        self.state = PipelineState()

        os.makedirs(output_dir, exist_ok=True)

        if not self.gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is required. Set it via environment or constructor."
            )

        # Initialize sub-agents
        self.research_agent = ResearchAgent(gemini_api_key=self.gemini_api_key)
        self.logging_agent = LoggingAgent(gemini_api_key=self.gemini_api_key)
        self.review_agent = CodeReviewAgent(gemini_api_key=self.gemini_api_key)
        self.fix_agent = CodeFixAgent(gemini_api_key=self.gemini_api_key)
        self.cpu_test_agent = CPUTestAgent(gemini_api_key=self.gemini_api_key)
        self.gpu_agent = GPUTrainingAgent(gemini_api_key=self.gemini_api_key)
        self.error_agent = ErrorAnalysisAgent(gemini_api_key=self.gemini_api_key)
        self.monitor_agent = MonitoringAgent(gemini_api_key=self.gemini_api_key)
        self._research_context = None

    def _get_research_context(self):
        """Get research context from ResearchAgent (cached after first call)."""
        if self._research_context is None:
            logger.info("ResearchAgent: Gathering best practices via Tavily...")
            self._research_context = self.research_agent.research(self.task)
            logger.info(f"ResearchAgent: {len(self._research_context)} chars of context gathered")
        return self._research_context

    def _query_gemini(self, prompt, system_prompt):
        """Query Gemini with high thinking for code generation."""
        return query_model(
            model_str=PIPELINE_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            gemini_api_key=self.gemini_api_key,
            print_cost=False,
        )

    # ─── Phase 1: Code Generation ────────────────────────────────────────

    def generate_training_code(self, for_gpu=False):
        """
        Generate training code using Gemini, incorporating research context
        and learnings from previous iterations.
        """
        device = "GPU (CUDA)" if for_gpu else "CPU"
        error_context = self.state.get_error_summary()
        history_context = self.state.get_history_summary()
        research = self._get_research_context()

        if for_gpu:
            device_requirements = (
                "GPU REQUIREMENTS:\n"
                '- Use torch.device("cuda") and move model/data to GPU\n'
                "- Use mixed precision (torch.cuda.amp) if beneficial\n"
                "- Use DataLoader with pin_memory=True and num_workers=2"
            )
        else:
            device_requirements = (
                "CPU REQUIREMENTS:\n"
                '- Use torch.device("cpu")\n'
                "- Keep model small and epochs low (1-2) for quick validation\n"
                "- This is a VALIDATION run, focus on correctness not performance"
            )

        system_prompt = (
            "You are an expert ML engineer writing Python training scripts for Kaggle notebooks. "
            "Write clean, complete, self-contained code that runs without modification. "
            "Always include proper error handling and progress logging. "
            "Use print() with flush=True for all output so logs appear in real-time. "
            "The code must be a complete script, not a notebook — no cell magic or !commands."
        )

        prompt = f"""Write a complete Python training script for the following task:

TASK: {self.task}

TARGET DEVICE: {device}
ITERATION: {self.state.iteration + 1}

{device_requirements}

BEST PRACTICES AND DOCUMENTATION:
{research[:2000]}

LOGGING REQUIREMENTS:
- Print progress every batch/epoch with loss values
- Print total training time at the end
- Print final metrics (accuracy, loss, etc.)
- Use print(..., flush=True) for all output

PREVIOUS ITERATION HISTORY:
{history_context}

ERRORS FROM PREVIOUS ITERATIONS (avoid these):
{error_context}

IMPORTANT:
- Write ONLY the Python code, no markdown, no explanations
- The script must be completely self-contained
- Import all dependencies at the top
- Save model/results to the current directory
- Handle errors gracefully with try/except
"""
        response = self._query_gemini(prompt, system_prompt)

        # Clean up response — remove markdown code fences if present
        code = response.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        return code

    # ─── Phase 2: Review & Fix (Sub-Agents) ──────────────────────────────

    def review_code(self, code, context=""):
        """
        Review code using CodeReviewAgent. If issues found, fix with CodeFixAgent.
        Returns (is_ok, fixed_code, review_notes).
        """
        research = self._get_research_context()
        device = "GPU" if "cuda" in code.lower() else "CPU"

        review = self.review_agent.review(code, research, device=device)

        if review["status"] == "OK":
            return True, code, review["notes"]

        # Issues found — use CodeFixAgent to fix
        logger.info(f"CodeReviewAgent found {len(review['issues'])} issues, "
                     "delegating to CodeFixAgent...")
        fixed_code = self.fix_agent.fix(
            code, review["issues"], research,
            error_history=self.state.get_error_summary()
        )
        return False, fixed_code, review["notes"]

    # ─── Phase 3: Analyze Logs (Sub-Agent) ───────────────────────────────

    def analyze_logs(self, logs, code):
        """
        Analyze logs using ErrorAnalysisAgent.
        Returns (has_issues, analysis_dict).
        """
        research = self._get_research_context()
        result = self.error_agent.analyze(logs, code, research)
        return result["has_errors"], result

    # ─── Phase 4: Submit and Monitor ─────────────────────────────────────

    def _save_code_file(self, code, name):
        """Save code to a file and return the path."""
        path = os.path.join(self.output_dir, name)
        with open(path, "w") as f:
            f.write(code)
        return path

    def submit_and_monitor(self, code, title, enable_gpu=False):
        """
        Submit code to Kaggle and monitor execution via log polling.
        Uses MonitoringAgent (Gemini 3 Flash) for real-time assessment.

        @param code: (str) Python code to execute
        @param title: (str) notebook title
        @param enable_gpu: (bool) whether to use GPU
        @return: (dict) with 'status', 'logs', 'errors', 'metrics'
        """
        # Inject structured logging (uses inject_logging_code for deterministic output)
        instrumented_code = inject_logging_code(code)

        # Save to file
        filename = f"train_iter{self.state.iteration}_{('gpu' if enable_gpu else 'cpu')}.py"
        code_path = self._save_code_file(instrumented_code, filename)

        logger.info(f"Submitting to Kaggle: {title} ({'GPU' if enable_gpu else 'CPU'})")

        # Submit
        kernel_ref = submit_training_notebook(
            title=title,
            code_file=code_path,
            enable_gpu=enable_gpu,
            dataset_sources=self.dataset_sources,
            competition_sources=self.competition_sources,
        )

        logger.info(f"Submitted kernel: {kernel_ref}")

        # Monitor via polling with MonitoringAgent assessment
        final_status = poll_notebook_status(
            kernel_ref,
            poll_interval=20,
            timeout=900 if enable_gpu else 600,
        )

        logger.info(f"Kernel finished: {final_status['status']} "
                     f"({final_status['elapsed']}s)")

        # Retrieve logs
        log_dir = os.path.join(self.output_dir, f"logs_iter{self.state.iteration}")
        logs_data = retrieve_notebook_logs(kernel_ref, output_dir=log_dir)
        parsed = parse_pipeline_logs(logs_data.get("log_content", ""))

        # MonitoringAgent (Gemini 3 Flash) provides lightweight assessment
        log_content = logs_data.get("log_content", "")
        if log_content:
            monitor_result = self.monitor_agent.assess_progress(
                log_content, self.task
            )
            logger.info(f"MonitoringAgent: {monitor_result['status']} — "
                         f"{monitor_result['message']}")
        else:
            monitor_result = {"status": "unknown", "message": "No logs available"}

        return {
            "status": final_status["status"],
            "kernel_ref": kernel_ref,
            "logs": log_content,
            "errors": parsed.get("errors", []) + logs_data.get("errors", []),
            "metrics": parsed.get("metrics", {}),
            "output_files": logs_data.get("output_files", []),
            "monitor_assessment": monitor_result,
        }

    # ─── Main Run Loop ───────────────────────────────────────────────────

    def run_iteration(self):
        """
        Run a single iteration of the training pipeline using sub-agents.

        Workflow:
            1. ResearchAgent → gather best practices context
            2. Generate code → LoggingAgent → CodeReviewAgent → CodeFixAgent
            3. CPUTestAgent → submit CPU test → MonitoringAgent watches
            4. ErrorAnalysisAgent → CodeFixAgent if issues
            5. GPUTrainingAgent → CodeReviewAgent → submit GPU training
            6. MonitoringAgent watches → summarize

        Returns (success, result_dict)
        """
        iteration = self.state.iteration
        logger.info(f"{'='*60}")
        logger.info(f"ITERATION {iteration + 1} — Sub-Agent Pipeline")
        logger.info(f"{'='*60}")

        # ── Step 1: ResearchAgent gathers context ──
        research = self._get_research_context()
        logger.info(f"Step 1: Research context ready ({len(research)} chars)")

        # ── Step 2: Generate training code ──
        logger.info("Step 2: Generating training code...")
        base_code = self.generate_training_code(for_gpu=False)
        self.state.record("generate", base_code)

        # ── Step 3: CodeReviewAgent reviews → CodeFixAgent fixes ──
        logger.info("Step 3: CodeReviewAgent reviewing code...")
        _, reviewed_code, review_notes = self.review_code(base_code)
        self.state.record("review", reviewed_code)

        # ── Step 4: CPUTestAgent creates CPU version ──
        logger.info("Step 4: CPUTestAgent creating CPU validation version...")
        cpu_code = self.cpu_test_agent.create_cpu_version(
            reviewed_code, research, self.task
        )
        self.state.record("cpu_adapt", cpu_code)

        # ── Step 5: Submit CPU test → MonitoringAgent watches ──
        logger.info("Step 5: Submitting CPU test to Kaggle...")
        cpu_title = f"pipeline-cpu-test-iter{iteration + 1}-{int(time.time())}"
        cpu_result = self.submit_and_monitor(
            cpu_code, cpu_title, enable_gpu=False
        )
        self.state.record("cpu_test", cpu_code,
                          logs=cpu_result["logs"],
                          errors=cpu_result["errors"],
                          metrics=cpu_result["metrics"])

        # ── Step 6: ErrorAnalysisAgent checks CPU results ──
        if cpu_result["errors"]:
            logger.warning(f"CPU test had {len(cpu_result['errors'])} errors!")
            has_issues, analysis = self.analyze_logs(
                cpu_result["logs"], cpu_code
            )
            self.state.record("analyze_cpu",
                              analysis.get("raw_analysis", str(analysis)),
                              errors=cpu_result["errors"])

            if has_issues and analysis.get("fixes"):
                # CodeFixAgent fixes the issues
                logger.info("CodeFixAgent fixing CPU test errors...")
                cpu_code = self.fix_agent.fix(
                    cpu_code, analysis["fixes"], research,
                    error_history=self.state.get_error_summary()
                )
                self.state.record("fix_cpu", cpu_code)
            else:
                logger.info("Errors found but no fixes identified, "
                            "will try next iteration")
                return False, cpu_result

        if cpu_result["status"] != "complete":
            logger.warning(f"CPU test did not complete: {cpu_result['status']}")
            return False, cpu_result

        logger.info("CPU test passed!")

        # ── Step 7: GPUTrainingAgent creates GPU version ──
        logger.info("Step 7: GPUTrainingAgent creating GPU training version...")
        gpu_code = self.gpu_agent.create_gpu_version(
            reviewed_code, research, self.task
        )
        self.state.record("gpu_adapt", gpu_code)

        # ── Step 8: CodeReviewAgent reviews GPU code ──
        logger.info("Step 8: CodeReviewAgent reviewing GPU code...")
        _, reviewed_gpu_code, gpu_notes = self.review_code(gpu_code)
        self.state.record("review_gpu", reviewed_gpu_code)

        # ── Step 9: Submit GPU training → MonitoringAgent watches ──
        logger.info("Step 9: Submitting GPU training to Kaggle...")
        gpu_title = f"pipeline-gpu-train-iter{iteration + 1}-{int(time.time())}"
        gpu_result = self.submit_and_monitor(
            reviewed_gpu_code, gpu_title, enable_gpu=True
        )
        self.state.record("gpu_train", reviewed_gpu_code,
                          logs=gpu_result["logs"],
                          errors=gpu_result["errors"],
                          metrics=gpu_result["metrics"])

        # ── Step 10: Analyze GPU results ──
        if gpu_result["errors"]:
            logger.warning(f"GPU training had {len(gpu_result['errors'])} errors")
            has_issues, analysis = self.analyze_logs(
                gpu_result["logs"], reviewed_gpu_code
            )
            self.state.record("analyze_gpu",
                              analysis.get("raw_analysis", str(analysis)),
                              errors=gpu_result["errors"])
            return False, gpu_result

        if gpu_result["status"] == "complete":
            # MonitoringAgent summarizes the run
            summary = self.monitor_agent.summarize_run(
                gpu_result["logs"], self.task
            )
            logger.info(f"MonitoringAgent run summary:\n{summary}")
            logger.info("GPU training completed successfully!")
            self.state.best_metrics = gpu_result["metrics"]
            return True, gpu_result

        return False, gpu_result

    def run(self, max_iterations=3):
        """
        Run the full self-iterating pipeline.

        @param max_iterations: (int) maximum number of iterations
        @return: (dict) final pipeline results
        """
        logger.info(f"Starting Kaggle Training Pipeline")
        logger.info(f"Task: {self.task}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Model: {PIPELINE_MODEL}")

        results = []
        for i in range(max_iterations):
            self.state.iteration = i

            try:
                success, result = self.run_iteration()
                results.append({
                    "iteration": i + 1,
                    "success": success,
                    "status": result.get("status"),
                    "errors": result.get("errors", []),
                    "metrics": result.get("metrics", {}),
                })

                if success:
                    logger.info(f"Pipeline succeeded at iteration {i + 1}!")
                    break
                else:
                    logger.info(f"Iteration {i + 1} had issues, trying next...")
            except Exception as e:
                logger.error(f"Iteration {i + 1} failed with exception: {e}")
                self.state.accumulated_errors.append(str(e))
                results.append({
                    "iteration": i + 1,
                    "success": False,
                    "status": "exception",
                    "errors": [str(e)],
                })

        # Final summary
        summary = {
            "task": self.task,
            "total_iterations": len(results),
            "final_success": any(r["success"] for r in results),
            "iterations": results,
            "error_summary": self.state.get_error_summary(),
        }

        summary_path = os.path.join(self.output_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Pipeline summary saved to {summary_path}")

        return summary


# ─── Standalone execution for testing ────────────────────────────────────

def run_pipeline_from_cli():
    """Entry point for running the pipeline from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Training Pipeline")
    parser.add_argument("--task", type=str, required=True,
                        help="Training task description")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum iterations")
    parser.add_argument("--dataset", type=str, action="append", default=[],
                        help="Kaggle dataset source (can specify multiple)")
    parser.add_argument("--output-dir", type=str,
                        default="./kaggle_pipeline_output",
                        help="Output directory")
    args = parser.parse_args()

    pipeline = KaggleTrainingPipeline(
        task_description=args.task,
        dataset_sources=args.dataset,
        output_dir=args.output_dir,
    )
    result = pipeline.run(max_iterations=args.max_iterations)

    if result["final_success"]:
        logger.info("Pipeline completed successfully!")
    else:
        logger.warning("Pipeline did not succeed within max iterations")

    return result


if __name__ == "__main__":
    run_pipeline_from_cli()
