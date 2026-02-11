"""
Kaggle Training Pipeline — Self-Iterating ML Training with Gemini 2.5 Pro.

This pipeline implements a self-reviewing, iterative training loop:

Phase 1 — CPU Test:   Generate training code → push to Kaggle CPU → monitor logs
Phase 2 — Review:     Analyze logs for errors → self-review with Gemini → fix
Phase 3 — GPU Train:  Adapt for GPU → push to Kaggle GPU → monitor training
Phase 4 — Iterate:    While monitoring, prepare next iteration (parallel work)

Key design principles:
- Real-time log monitoring prevents waiting 10 min for hidden errors
- Gemini 2.5 Pro with high thinking reviews code before each push
- Each iteration learns from previous failures via log analysis
- Parallel preparation: next iteration code is ready before current completes

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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Gemini model used for code generation and self-review
PIPELINE_MODEL = "gemini-2.5-pro"


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
    Self-iterating ML training pipeline using Kaggle notebooks and Gemini 2.5 Pro.

    Workflow per iteration:
    1. Generate/refine training code using Gemini (learns from past errors)
    2. Inject logging → push to Kaggle CPU → monitor logs → catch errors early
    3. If CPU test passes, adapt for GPU → push to Kaggle GPU → monitor training
    4. While GPU training runs, pre-generate next iteration code (parallel)
    5. Analyze results → feed back into next iteration
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

    def _query_gemini(self, prompt, system_prompt):
        """Query Gemini 2.5 Pro with high thinking for code generation/review."""
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
        Use Gemini 2.5 Pro to generate training code.
        Incorporates learnings from previous iterations.
        """
        device = "GPU (CUDA)" if for_gpu else "CPU"
        error_context = self.state.get_error_summary()
        history_context = self.state.get_history_summary()

        system_prompt = (
            "You are an expert ML engineer writing Python training scripts for Kaggle notebooks. "
            "Write clean, complete, self-contained code that runs without modification. "
            "Always include proper error handling and progress logging. "
            "Use print() with flush=True for all output so logs appear in real-time. "
            "The code must be a complete script, not a notebook — no cell magic or !commands."
        )

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

        prompt = f"""Write a complete Python training script for the following task:

TASK: {self.task}

TARGET DEVICE: {device}
ITERATION: {self.state.iteration + 1}

{device_requirements}

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

    # ─── Phase 2: Self-Review ────────────────────────────────────────────

    def review_code(self, code, context=""):
        """
        Use Gemini 2.5 Pro to self-review code before submission.
        Returns (is_ok, fixed_code, review_notes).
        """
        system_prompt = (
            "You are a senior ML engineer reviewing Python training code for Kaggle. "
            "Check for: import errors, syntax errors, runtime errors, device mismatches, "
            "missing dependencies, infinite loops, memory issues, and Kaggle-specific "
            "constraints (no internet for datasets after start, 9h GPU / 12h CPU limits). "
            "If the code has issues, fix them and return the corrected code. "
            "If the code is correct, return it unchanged."
        )

        prompt = f"""Review this training code and fix any issues:

{context}

```python
{code}
```

RESPOND WITH EXACTLY:
1. First line: REVIEW_STATUS: OK or REVIEW_STATUS: FIXED
2. Second line: REVIEW_NOTES: <brief description of any changes>
3. Then the complete corrected code between ```python and ``` markers

If the code is fine, still include it in full after the markers.
"""
        response = self._query_gemini(prompt, system_prompt)

        is_ok = "REVIEW_STATUS: OK" in response
        notes = ""
        fixed_code = code  # default to original

        for line in response.split("\n"):
            if line.startswith("REVIEW_NOTES:"):
                notes = line.replace("REVIEW_NOTES:", "").strip()
                break

        # Extract code from response
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0].strip()
                if code_part:
                    fixed_code = code_part

        logger.info(f"Code review: {'PASS' if is_ok else 'FIXED'} — {notes}")
        return is_ok, fixed_code, notes

    # ─── Phase 3: Analyze Logs ───────────────────────────────────────────

    def analyze_logs(self, logs, code):
        """
        Use Gemini 2.5 Pro to analyze training logs and identify issues.
        Returns (has_issues, analysis, suggested_fixes).
        """
        system_prompt = (
            "You are an ML debugging expert. Analyze training logs from a Kaggle notebook. "
            "Identify: errors, warnings, performance issues, convergence problems. "
            "Be specific about what went wrong and how to fix it."
        )

        prompt = f"""Analyze these training logs and identify any issues:

LOGS:
{logs[:3000]}

CODE THAT PRODUCED THESE LOGS:
```python
{code[:2000]}
```

RESPOND WITH:
1. First line: LOG_STATUS: OK or LOG_STATUS: ISSUES_FOUND
2. ANALYSIS: <detailed analysis>
3. FIXES: <specific code changes needed, if any>
"""
        response = self._query_gemini(prompt, system_prompt)

        has_issues = "LOG_STATUS: ISSUES_FOUND" in response
        return has_issues, response

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

        @param code: (str) Python code to execute
        @param title: (str) notebook title
        @param enable_gpu: (bool) whether to use GPU
        @return: (dict) with 'status', 'logs', 'errors', 'metrics'
        """
        # Inject logging code
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

        # Monitor via polling
        final_status = poll_notebook_status(
            kernel_ref,
            poll_interval=20,
            timeout=900 if enable_gpu else 600,  # 15 min GPU, 10 min CPU
        )

        logger.info(f"Kernel finished: {final_status['status']} "
                     f"({final_status['elapsed']}s)")

        # Retrieve logs
        log_dir = os.path.join(self.output_dir, f"logs_iter{self.state.iteration}")
        logs_data = retrieve_notebook_logs(kernel_ref, output_dir=log_dir)
        parsed = parse_pipeline_logs(logs_data.get("log_content", ""))

        return {
            "status": final_status["status"],
            "kernel_ref": kernel_ref,
            "logs": logs_data.get("log_content", ""),
            "errors": parsed.get("errors", []) + logs_data.get("errors", []),
            "metrics": parsed.get("metrics", {}),
            "output_files": logs_data.get("output_files", []),
        }

    # ─── Main Run Loop ───────────────────────────────────────────────────

    def run_iteration(self):
        """
        Run a single iteration of the training pipeline.

        Returns (success, result_dict)
        """
        iteration = self.state.iteration
        logger.info(f"{'='*60}")
        logger.info(f"ITERATION {iteration + 1}")
        logger.info(f"{'='*60}")

        # ── Step 1: Generate CPU test code ──
        logger.info("Step 1: Generating CPU test code with Gemini 2.5 Pro...")
        cpu_code = self.generate_training_code(for_gpu=False)
        self.state.record("generate_cpu", cpu_code)
        logger.info(f"Generated {len(cpu_code)} chars of CPU code")

        # ── Step 2: Self-review CPU code ──
        logger.info("Step 2: Self-reviewing CPU code...")
        _, reviewed_cpu_code, review_notes = self.review_code(
            cpu_code,
            context="CPU validation run — keep it fast, focus on correctness"
        )
        self.state.record("review_cpu", reviewed_cpu_code)

        # ── Step 3: Submit CPU test ──
        logger.info("Step 3: Submitting CPU test to Kaggle...")
        cpu_title = f"pipeline-cpu-test-iter{iteration + 1}-{int(time.time())}"
        cpu_result = self.submit_and_monitor(
            reviewed_cpu_code, cpu_title, enable_gpu=False
        )
        self.state.record("cpu_test", reviewed_cpu_code,
                          logs=cpu_result["logs"],
                          errors=cpu_result["errors"],
                          metrics=cpu_result["metrics"])

        # ── Step 4: Analyze CPU logs ──
        if cpu_result["errors"]:
            logger.warning(f"CPU test had {len(cpu_result['errors'])} errors!")
            has_issues, analysis = self.analyze_logs(
                cpu_result["logs"], reviewed_cpu_code
            )
            self.state.record("analyze_cpu_errors", analysis,
                              errors=cpu_result["errors"])

            if has_issues:
                logger.info("Errors found in CPU test, will fix in next iteration")
                return False, cpu_result

        if cpu_result["status"] != "complete":
            logger.warning(f"CPU test did not complete: {cpu_result['status']}")
            return False, cpu_result

        logger.info("CPU test passed!")

        # ── Step 5: Generate GPU code (parallel-ready) ──
        logger.info("Step 5: Generating GPU training code...")
        gpu_code = self.generate_training_code(for_gpu=True)
        self.state.record("generate_gpu", gpu_code)

        # ── Step 6: Self-review GPU code ──
        logger.info("Step 6: Self-reviewing GPU code...")
        _, reviewed_gpu_code, gpu_notes = self.review_code(
            gpu_code,
            context="GPU training run — use CUDA, mixed precision, proper data loading"
        )
        self.state.record("review_gpu", reviewed_gpu_code)

        # ── Step 7: Submit GPU training ──
        logger.info("Step 7: Submitting GPU training to Kaggle...")
        gpu_title = f"pipeline-gpu-train-iter{iteration + 1}-{int(time.time())}"

        # Start preparing next iteration while GPU runs (parallel work)
        next_code = None
        next_thread = None
        if self.state.iteration < 10:  # safety cap
            next_iter = self.state.iteration + 1
            def _prepare_next():
                nonlocal next_code
                try:
                    # Use a snapshot of iteration, not shared mutable state
                    saved = self.state.iteration
                    self.state.iteration = next_iter
                    next_code = self.generate_training_code(for_gpu=False)
                    self.state.iteration = saved
                except Exception as e:
                    logger.warning(f"Pre-generation failed: {e}")

            next_thread = threading.Thread(target=_prepare_next, daemon=True)
            next_thread.start()
            logger.info("Started pre-generating next iteration code (parallel)")

        gpu_result = self.submit_and_monitor(
            reviewed_gpu_code, gpu_title, enable_gpu=True
        )
        self.state.record("gpu_train", reviewed_gpu_code,
                          logs=gpu_result["logs"],
                          errors=gpu_result["errors"],
                          metrics=gpu_result["metrics"])

        # Wait for parallel preparation if running
        if next_thread and next_thread.is_alive():
            next_thread.join(timeout=30)

        if next_code:
            self.state.best_code = next_code
            logger.info("Next iteration code pre-generated and ready")

        # ── Step 8: Analyze GPU results ──
        if gpu_result["errors"]:
            logger.warning(f"GPU training had {len(gpu_result['errors'])} errors")
            has_issues, analysis = self.analyze_logs(
                gpu_result["logs"], reviewed_gpu_code
            )
            self.state.record("analyze_gpu_errors", analysis,
                              errors=gpu_result["errors"])
            return False, gpu_result

        if gpu_result["status"] == "complete":
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
