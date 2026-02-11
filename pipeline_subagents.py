"""
Pipeline Sub-Agents for the Kaggle Training Pipeline.

Each sub-agent handles a specific phase of the ML training workflow.
The ResearchAgent uses Tavily to gather best practices and documentation,
which is then passed as context to all other agents.

Sub-agents:
    ResearchAgent       — Uses Tavily to search for Kaggle best practices, GPU docs,
                          environment docs, and sample code. Provides context to all others.
    LoggingAgent        — Injects logging and monitoring into training code.
    CodeReviewAgent     — Reviews code for errors, Kaggle constraints, best practices.
    CodeFixAgent        — Fixes issues identified by CodeReviewAgent or ErrorAnalysisAgent.
    CPUTestAgent        — Adapts code for CPU-only validation run.
    GPUTrainingAgent    — Adapts code for full GPU training with acceleration best practices.
    ErrorAnalysisAgent  — Analyzes execution logs/errors and produces diagnosis.
    MonitoringAgent     — Lightweight agent (Gemini 2.5 Flash) for real-time log monitoring.
"""

import os
import logging

from inference import query_model

logger = logging.getLogger(__name__)

# Models used by sub-agents
THINKING_MODEL = "gemini-2.5-pro"       # For code generation, review, fixing
MONITORING_MODEL = "gemini-2.5-flash"   # For lightweight monitoring


def _query(model_str, prompt, system_prompt, gemini_api_key=None):
    """Shared query helper for all sub-agents."""
    key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    return query_model(
        model_str=model_str,
        prompt=prompt,
        system_prompt=system_prompt,
        gemini_api_key=key,
        print_cost=False,
    )


def _extract_code(response):
    """Extract Python code from an LLM response that may contain markdown fences."""
    code = response.strip()
    if "```python" in code:
        parts = code.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


# ─── ResearchAgent (Tavily) ─────────────────────────────────────────────

class ResearchAgent:
    """
    Uses Tavily web search to gather best practices, documentation, and
    sample code relevant to the current training task and Kaggle environment.
    Falls back to the static kaggle_best_practices.md if Tavily is unavailable.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key
        self._cache = {}

    def _tavily_search(self, query, max_results=5):
        """Search the web using Tavily API."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set, skipping web search")
            return []

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=max_results,
                                     search_depth="basic")
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],
                })
            return results
        except ImportError:
            logger.warning("tavily-python not installed, skipping web search")
            return []
        except Exception as e:
            logger.warning(f"Tavily search failed for query \"{query}\": {e}")
            return []

    def _load_static_fallback(self):
        """Load static best practices as fallback."""
        md_path = os.path.join(os.path.dirname(__file__), "kaggle_best_practices.md")
        if os.path.exists(md_path):
            with open(md_path, "r") as f:
                return f.read()
        return ""

    def research(self, task_description, topics=None):
        """
        Gather best practices and documentation for the given task.

        @param task_description: (str) what the training task is
        @param topics: (list, optional) specific topics to research
        @return: (str) compiled research context for other agents
        """
        cache_key = task_description[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        if topics is None:
            topics = [
                "Kaggle notebook GPU training best practices",
                "Kaggle Python environment pre-installed packages",
                "PyTorch mixed precision training Kaggle",
                "Kaggle notebook logging monitoring progress",
            ]

        all_results = []
        for topic in topics:
            results = self._tavily_search(f"{topic} {task_description}")
            all_results.extend(results)

        if all_results:
            # Synthesize results using Gemini
            search_context = "\n\n".join(
                f"### {r['title']}\nSource: {r['url']}\n{r['content']}"
                for r in all_results
            )

            system_prompt = (
                "You are a research assistant that synthesizes web search results "
                "into actionable best practices for ML training on Kaggle. "
                "Focus on: logging patterns, GPU optimization, environment setup, "
                "common pitfalls, and code patterns."
            )
            prompt = f"""Synthesize these search results into a concise best practices
guide for the following task:

TASK: {task_description}

SEARCH RESULTS:
{search_context[:4000]}

Produce a concise guide with:
1. Environment setup (pre-installed packages, paths, limits)
2. Logging best practices (what to log, format, flush)
3. GPU optimization (mixed precision, DataLoader, memory)
4. Common pitfalls and how to avoid them
5. Sample code patterns
"""
            context = _query(THINKING_MODEL, prompt, system_prompt,
                             self.gemini_api_key)
        else:
            # Fall back to static best practices
            logger.info("No Tavily results, using static best practices")
            context = self._load_static_fallback()

        if not context:
            context = "No best practices available. Use standard ML training patterns."

        self._cache[cache_key] = context
        logger.info(f"ResearchAgent gathered {len(all_results)} web results, "
                     f"produced {len(context)} chars of context")
        return context


# ─── LoggingAgent ────────────────────────────────────────────────────────

class LoggingAgent:
    """
    Injects logging and monitoring code into training scripts based on
    best practices from the ResearchAgent.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def inject_logging(self, code, research_context, task_description):
        """
        Add structured logging to a training script.

        @param code: (str) original training code
        @param research_context: (str) best practices from ResearchAgent
        @param task_description: (str) what the training task is
        @return: (str) code with logging injected
        """
        system_prompt = (
            "You are an expert at adding production-quality logging to ML training scripts. "
            "Add structured logging that enables real-time monitoring. "
            "Do NOT change the training logic — only add logging. "
            "Use print(..., flush=True) for all output. "
            "Add: startup info, epoch/batch progress, metrics, error handling, timing."
        )

        prompt = f"""Add logging and monitoring to this training script.

TASK: {task_description}

BEST PRACTICES TO FOLLOW:
{research_context[:2000]}

ORIGINAL CODE:
```python
{code}
```

REQUIREMENTS:
- Add structured log lines with format: [PIPELINE LEVEL timestamp] message
- Log system info at startup (Python, PyTorch, CUDA, GPU name/memory)
- Log epoch progress with loss, accuracy, timing
- Log GPU memory usage periodically
- Wrap entire script in try/except for error capture
- Do NOT change training logic, only add logging
- Return the COMPLETE modified code

Respond with ONLY the complete Python code, no markdown or explanation.
"""
        response = _query(THINKING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)
        return _extract_code(response)


# ─── CodeReviewAgent ─────────────────────────────────────────────────────

class CodeReviewAgent:
    """
    Reviews training code for errors, Kaggle constraints, and best practices.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def review(self, code, research_context, device="CPU"):
        """
        Review code and return structured feedback.

        @param code: (str) code to review
        @param research_context: (str) best practices from ResearchAgent
        @param device: (str) target device (CPU or GPU)
        @return: (dict) with 'status' (OK|ISSUES), 'issues' (list), 'notes' (str)
        """
        system_prompt = (
            "You are a senior ML engineer reviewing Python code for Kaggle notebooks. "
            "Check for: syntax errors, import errors, runtime errors, device mismatches, "
            "missing dependencies, infinite loops, memory issues, Kaggle constraints "
            "(9h GPU / 12h CPU limit, file paths, internet access). "
            "Be specific about issues and their line locations."
        )

        prompt = f"""Review this {device} training code for a Kaggle notebook.

BEST PRACTICES:
{research_context[:1500]}

CODE:
```python
{code}
```

Respond in this exact format:
REVIEW_STATUS: OK or REVIEW_STATUS: ISSUES
ISSUES:
- <issue 1 with line number>
- <issue 2 with line number>
REVIEW_NOTES: <summary>
"""
        response = _query(THINKING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)

        status = "OK" if "REVIEW_STATUS: OK" in response else "ISSUES"
        issues = []
        notes = ""

        in_issues = False
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ISSUES:"):
                in_issues = True
                continue
            if line.startswith("REVIEW_NOTES:"):
                in_issues = False
                notes = line.replace("REVIEW_NOTES:", "").strip()
                continue
            if in_issues and line.startswith("- "):
                issues.append(line[2:])

        logger.info(f"CodeReview [{device}]: {status} — {len(issues)} issues — {notes}")
        return {"status": status, "issues": issues, "notes": notes}


# ─── CodeFixAgent ────────────────────────────────────────────────────────

class CodeFixAgent:
    """
    Fixes issues in code based on review feedback or error analysis.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def fix(self, code, issues, research_context, error_history=None):
        """
        Fix identified issues in code.

        @param code: (str) code with issues
        @param issues: (list) issues to fix
        @param research_context: (str) best practices from ResearchAgent
        @param error_history: (str, optional) past errors to avoid
        @return: (str) fixed code
        """
        issues_str = "\n".join(f"- {i}" for i in issues)
        history_str = error_history or "No previous errors."

        system_prompt = (
            "You are an expert ML engineer fixing Python training code. "
            "Fix ALL identified issues. Keep the training logic intact. "
            "Ensure the code is complete and self-contained."
        )

        prompt = f"""Fix these issues in the training code:

ISSUES TO FIX:
{issues_str}

PREVIOUS ERRORS TO AVOID:
{history_str}

BEST PRACTICES:
{research_context[:1500]}

CODE TO FIX:
```python
{code}
```

Return the COMPLETE fixed code. Do not leave anything out.
Respond with ONLY the Python code, no markdown or explanation.
"""
        response = _query(THINKING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)
        return _extract_code(response)


# ─── CPUTestAgent ────────────────────────────────────────────────────────

class CPUTestAgent:
    """
    Creates a CPU-only version of training code for quick validation
    before committing GPU quota.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def create_cpu_version(self, code, research_context, task_description):
        """
        Create a CPU-only validation version of the training code.

        @param code: (str) original training code
        @param research_context: (str) best practices from ResearchAgent
        @param task_description: (str) what the training task is
        @return: (str) CPU-only version of the code
        """
        system_prompt = (
            "You are an expert at adapting ML training code for CPU-only validation. "
            "The goal is a fast smoke test that validates correctness without GPU. "
            "Keep all logic identical, just reduce scale."
        )

        prompt = f"""Create a CPU-only validation version of this training code.

TASK: {task_description}

BEST PRACTICES:
{research_context[:1500]}

ORIGINAL CODE:
```python
{code}
```

REQUIREMENTS:
- Force device = "cpu" — remove all CUDA references
- Reduce epochs to 1-2
- Reduce batch size if needed for speed
- Use a small subset of data (first 100-500 samples)
- Keep ALL training logic and logging intact
- Must run in under 5 minutes on CPU
- Keep it complete and self-contained

Return the COMPLETE CPU version. Respond with ONLY Python code.
"""
        response = _query(THINKING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)
        return _extract_code(response)


# ─── GPUTrainingAgent ────────────────────────────────────────────────────

class GPUTrainingAgent:
    """
    Creates the full GPU-accelerated training code with optimizations
    based on Kaggle GPU documentation and best practices.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def create_gpu_version(self, code, research_context, task_description):
        """
        Create a full GPU-accelerated training version.

        @param code: (str) validated CPU training code
        @param research_context: (str) best practices from ResearchAgent
        @param task_description: (str) what the training task is
        @return: (str) GPU-optimized version of the code
        """
        system_prompt = (
            "You are an expert at GPU-accelerated ML training on Kaggle. "
            "Optimize for Kaggle's P100/T4 GPUs with 16GB VRAM. "
            "Apply all GPU best practices: mixed precision, efficient data loading, "
            "memory management, proper checkpointing."
        )

        prompt = f"""Create a full GPU-accelerated version of this training code for Kaggle.

TASK: {task_description}

KAGGLE GPU BEST PRACTICES:
{research_context[:2000]}

CPU-VALIDATED CODE:
```python
{code}
```

REQUIREMENTS:
- Use torch.device("cuda") with fallback to "cpu"
- Apply mixed precision training (torch.cuda.amp)
- Use DataLoader with pin_memory=True, num_workers=2
- Add GPU memory monitoring and logging
- Add checkpoint saving every N epochs
- Full training epochs (not reduced like CPU test)
- Keep all logging and monitoring intact
- Handle torch.cuda.OutOfMemoryError gracefully
- Clear CUDA cache between epochs
- Must be complete and self-contained

Return the COMPLETE GPU-optimized code. Respond with ONLY Python code.
"""
        response = _query(THINKING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)
        return _extract_code(response)


# ─── ErrorAnalysisAgent ──────────────────────────────────────────────────

class ErrorAnalysisAgent:
    """
    Analyzes execution logs and errors to produce a diagnosis and
    recommended fixes.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def analyze(self, logs, code, research_context):
        """
        Analyze logs and identify issues.

        @param logs: (str) execution logs
        @param code: (str) code that produced the logs
        @param research_context: (str) best practices from ResearchAgent
        @return: (dict) with 'has_errors' (bool), 'diagnosis' (str),
                 'fixes' (list of str), 'severity' (str)
        """
        system_prompt = (
            "You are an ML debugging expert. Analyze training logs and identify: "
            "errors, warnings, performance issues, convergence problems, OOM issues. "
            "Be specific about root causes and fixes."
        )

        prompt = f"""Analyze these training logs and identify issues.

BEST PRACTICES:
{research_context[:1000]}

LOGS (last 3000 chars):
{logs[-3000:]}

CODE:
```python
{code[:2000]}
```

Respond in this format:
ERROR_STATUS: NONE or ERROR_STATUS: FOUND
SEVERITY: low|medium|high|critical
DIAGNOSIS: <detailed analysis of what went wrong>
FIXES:
- <specific fix 1>
- <specific fix 2>
"""
        response = _query(THINKING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)

        has_errors = "ERROR_STATUS: FOUND" in response
        severity = "low"
        diagnosis = ""
        fixes = []

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SEVERITY:"):
                severity = line.replace("SEVERITY:", "").strip().lower()
            elif line.startswith("DIAGNOSIS:"):
                diagnosis = line.replace("DIAGNOSIS:", "").strip()
            elif line.startswith("- "):
                fixes.append(line[2:])

        logger.info(f"ErrorAnalysis: errors={has_errors}, severity={severity}")
        return {
            "has_errors": has_errors,
            "diagnosis": diagnosis,
            "fixes": fixes,
            "severity": severity,
            "raw_analysis": response,
        }


# ─── MonitoringAgent (Gemini 2.5 Flash) ──────────────────────────────────

class MonitoringAgent:
    """
    Lightweight agent using Gemini 2.5 Flash for real-time training
    progress monitoring. Parses logs and provides quick status assessments.
    """

    def __init__(self, gemini_api_key=None):
        self.gemini_api_key = gemini_api_key

    def assess_progress(self, logs, task_description):
        """
        Quick assessment of training progress from recent logs.

        @param logs: (str) recent log content
        @param task_description: (str) what the training task is
        @return: (dict) with 'status' (healthy|warning|error),
                 'message' (str), 'should_continue' (bool)
        """
        system_prompt = (
            "You are a training monitor. Quickly assess if training is progressing "
            "normally. Look for: NaN loss, diverging loss, OOM errors, stalled "
            "progress, import errors. Respond concisely."
        )

        prompt = f"""Quick status check on this training run:

TASK: {task_description}

RECENT LOGS:
{logs[-2000:]}

Respond in exactly this format:
STATUS: healthy|warning|error
MESSAGE: <one line summary>
SHOULD_CONTINUE: yes|no
"""
        response = _query(MONITORING_MODEL, prompt, system_prompt,
                          self.gemini_api_key)

        status = "healthy"
        message = ""
        should_continue = True

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("STATUS:"):
                status = line.replace("STATUS:", "").strip().lower()
            elif line.startswith("MESSAGE:"):
                message = line.replace("MESSAGE:", "").strip()
            elif line.startswith("SHOULD_CONTINUE:"):
                should_continue = "yes" in line.lower()

        logger.info(f"Monitor: {status} — {message}")
        return {
            "status": status,
            "message": message,
            "should_continue": should_continue,
        }

    def summarize_run(self, full_logs, task_description):
        """
        Produce a final summary of a completed training run.

        @param full_logs: (str) complete log content
        @param task_description: (str) what the training task is
        @return: (str) human-readable summary
        """
        system_prompt = (
            "You are a training run summarizer. Produce a brief summary of the "
            "training run including: final metrics, training time, any issues, "
            "and recommendations for next iteration."
        )

        prompt = f"""Summarize this training run:

TASK: {task_description}

FULL LOGS:
{full_logs[-4000:]}

Provide a brief summary with: final metrics, total time, issues found,
and recommendations for improvement.
"""
        return _query(MONITORING_MODEL, prompt, system_prompt,
                      self.gemini_api_key)
