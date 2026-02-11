# Cross-Analysis: AgentLaboratory vs AI-Researcher

Deep cross-analysis of AgentLaboratory with AI-Researcher
(https://github.com/HKUDS/AI-Researcher, NeurIPS 2025 Spotlight,
arXiv:2502.05957) and other related projects. This document identifies
gaps, architectural advantages, and actionable improvements.

## Reference Projects

- [HKUDS/AI-Researcher](https://github.com/HKUDS/AI-Researcher) — NeurIPS 2025 Spotlight,
  full autonomous research system with Docker-based sandboxed execution
- [SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory) — upstream project
- [romgenie/AgentLaboratoryReinvented](https://github.com/romgenie/AgentLaboratoryReinvented) — enhanced fork
- [allenai/SAGE](https://github.com/allenai/SAGE) — Gemini integration with caching
- [ExtensityAI/symbolicai](https://github.com/ExtensityAI/symbolicai) — Gemini reasoning engine

## Architecture Comparison: AgentLaboratory vs AI-Researcher

### Agent Systems

| Feature | AgentLaboratory | AI-Researcher |
|---------|----------------|---------------|
| **Agent roles** | PhDStudent, MLEngineer, SWEngineer, Postdoc, Professor, Reviewers | SurveyAgent, IdeaAgent, PlanAgent, MLAgent, PrepareAgent, ExpAnalyser, JudgeAgent |
| **Code execution** | Local multiprocessing with 600s timeout | Docker-sandboxed execution via TCP/port |
| **GPU execution** | Local only (now + Kaggle via pipeline) | Docker with GPU passthrough (`"device=0"`) |
| **Workflow** | Linear phases: lit review → plan → data prep → experiments → results → report | DAG-based FlowGraph with FlowCache for state |
| **Self-review** | Reviewer agents score final report | JudgeAgent evaluates at each iteration |
| **Iteration** | Single pass with optional report refinement loop | Continuous iteration with max_iter_times control |
| **Memory** | Conversation history in list | FlowCache with persistent state management |
| **LLM backend** | OpenAI, Gemini, DeepSeek, Anthropic | LiteLLM (any provider via OpenRouter) |

### Key Advantages of AI-Researcher (that AgentLaboratory lacks)

1. **Docker-sandboxed execution**: AI-Researcher runs ALL generated code inside
   Docker containers, preventing system damage and ensuring reproducibility.
   AgentLaboratory runs code directly on the host.

2. **DAG-based workflow (FlowGraph)**: AI-Researcher uses a directed acyclic
   graph for workflow execution with `FlowCache` for caching intermediate
   results. AgentLaboratory uses a rigid linear phase sequence.

3. **JudgeAgent for continuous evaluation**: AI-Researcher has a dedicated
   `JudgeAgent` that evaluates experiment results and decides whether to
   iterate, providing quality gates at each step. AgentLaboratory only
   evaluates the final report via ReviewersAgent.

4. **Experiment Analysis Agent**: AI-Researcher's `ExpAnalyser` deeply
   analyzes experimental results to identify improvements. AgentLaboratory's
   results interpretation phase is less structured.

5. **Dual completion models**: AI-Researcher uses `COMPLETION_MODEL` for
   complex reasoning and `CHEEP_MODEL` (their naming for a cheap/fast model)
   for simpler tasks, reducing cost. AgentLaboratory uses a single model per
   phase.

6. **Structured benchmark system**: AI-Researcher includes a benchmark
   suite for evaluating the system across categories (GNN, VQ, diffusion,
   recommendation, reasoning).

7. **Web GUI (Gradio)**: AI-Researcher provides a web interface for
   configuration and monitoring. AgentLaboratory has a Streamlit app but
   it's more limited.

### Key Advantages of AgentLaboratory (that AI-Researcher lacks)

1. **Kaggle integration**: Remote GPU execution via Kaggle notebooks with
   real-time log monitoring — doesn't require local GPU hardware.

2. **Sub-agent pipeline**: Specialized sub-agents (ResearchAgent, CPUTestAgent,
   GPUTrainingAgent, MonitoringAgent) for the training pipeline.

3. **Tavily-powered research**: Dynamic web search for best practices using
   Tavily API, not just static reference papers.

4. **Gemini 3 Pro with max thinking**: Extended thinking mode with
   `thinking_budget=24576` for deeper reasoning.

5. **Multi-LLM support without LiteLLM**: Direct integration with OpenAI,
   Gemini, DeepSeek, Anthropic without an intermediary library.

## Improvements Already Implemented

- **Environment variable management**: `.env.example` template for all API keys
- **Docker support**: Dockerfile using Kaggle's Python Docker image
- **Kaggle API integration**: `kaggle_utils.py` for remote GPU training
- **Gemini 3 Pro with max thinking**: `ThinkingConfig(thinking_budget=24576)`
- **Gemini-only key support**: No longer requires OpenAI or Anthropic
- **Sub-agent architecture**: 8 specialized pipeline sub-agents
- **Tavily research agent**: Dynamic best practices gathering
- **Monitoring agent**: Gemini 3 Flash for real-time training monitoring

## Actionable Improvements (Inspired by AI-Researcher)

### Priority 1: Sandboxed Code Execution

**Problem**: AgentLaboratory runs generated code directly on the host machine.
AI-Researcher sandboxes everything in Docker containers via TCP.

**Implementation**: Use the existing Dockerfile and docker-compose.yml to
execute `run_experiments.py` inside a container. The `execute_code()` function
in `tools.py` should optionally route to Docker execution.

### Priority 2: JudgeAgent for Iterative Quality Gates

**Problem**: AgentLaboratory only evaluates the final report. AI-Researcher's
`JudgeAgent` evaluates experiments at each iteration and decides whether
results are good enough to proceed or need another round.

**Implementation**: Add a `JudgeAgent` to the existing agent system that runs
after each experiment, scoring results and deciding whether to iterate or
proceed. This prevents wasting compute on poor experiment paths.

### Priority 3: Dual Model Strategy (Expensive + Cheap)

**Problem**: AgentLaboratory uses the same model for all tasks in a phase.
AI-Researcher uses `COMPLETION_MODEL` for complex tasks and a cheaper model
for simple ones (e.g., formatting, summarization).

**Implementation**: Add a `cheap_model` parameter alongside the existing
model backbone. Use it for: log summarization, simple formatting, status
checks, and monitoring tasks. Route expensive tasks (code generation,
review, planning) to the full model.

### Priority 4: FlowGraph-based Workflow

**Problem**: AgentLaboratory has a rigid linear phase sequence. If
experiments fail, it restarts from plan formulation. AI-Researcher's
FlowGraph allows flexible DAG execution with caching.

**Implementation**: Replace the linear phase loop in `ai_lab_repo.py`
with a configurable workflow graph. Allow phases to be re-run selectively
without restarting the entire pipeline.

### Priority 5: Response Caching (FlowCache)

**Problem**: Repeated LLM calls during iteration waste tokens and money.
AI-Researcher's `FlowCache` caches intermediate results.

**Implementation**: Add a caching layer to `inference.py` that hashes
(model + system_prompt + prompt) and returns cached responses for
identical queries. Use file-based cache with configurable TTL.

### Priority 6: Structured Result Analysis

**Problem**: AgentLaboratory's results interpretation is freeform.
AI-Researcher's `ExpAnalyser` agent uses structured analysis templates.

**Implementation**: Add structured analysis prompts to the results
interpretation phase that extract: key metrics, comparisons to baselines,
failure modes, and specific improvement suggestions.

### Priority 7: Gemini Cost Tracking

**Problem**: AgentLaboratory tracks costs for OpenAI models but not Gemini.

**Implementation**: Add token counting for Gemini API responses using
`response.usage_metadata` and calculate costs based on Gemini pricing.

### Priority 8: Environment Capabilities System

**Problem**: Agents don't know what compute resources are available.

**Implementation**: Add a capabilities discovery system that detects:
available GPUs, memory, installed packages, Kaggle availability, and
Docker availability. Pass this context to agents for better planning.
