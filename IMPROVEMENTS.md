# Cross-Analysis: AgentLaboratory vs HKUDS Research Ecosystem

Deep cross-analysis of AgentLaboratory with the full HKUDS autonomous
research ecosystem and related projects. This document identifies gaps,
architectural advantages, and actionable improvements.

## Reference Projects Analyzed

### HKUDS Ecosystem (HKU Data Science Lab)
- [HKUDS/AI-Researcher](https://github.com/HKUDS/AI-Researcher) — NeurIPS 2025 Spotlight
  (arXiv:2502.05957). Docker-sandboxed code execution, FlowGraph DAG
  workflow, JudgeAgent quality gates. 7 specialized research agents.
- [HKUDS/AutoAgent](https://github.com/HKUDS/AutoAgent) — Meta-agent
  framework. Agents that dynamically CREATE other agents/tools/workflows.
  Event-driven flow engine with async execution. Agent/Tool/Workflow
  registry for runtime composition.
- [HKUDS/Auto-Deep-Research](https://github.com/HKUDS/Auto-Deep-Research)
  — Automated deep research pipeline with file selection, web search,
  and multi-agent collaboration. Built on AutoAgent framework.
- [HKUDS/DeepCode](https://github.com/HKUDS/DeepCode) — AI research
  engine that transforms papers into working code. 78KB orchestration
  engine, MCP-based tool servers, Docker sandboxing via nanobot,
  FastAPI+React UI.
- [HKUDS/DeepResearch-Eval](https://github.com/HKUDS/DeepResearch-Eval)
  — Research quality evaluation with fact-checking (`judge_fact.py`) and
  multi-dimension scoring (`judge_score.py`). Structured evaluation prompts.

### Other Reference Projects
- [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 50k+ stars,
  Role-Action-Message pattern, shared Environment with pub/sub messaging,
  Team orchestration with budget control, Experience Pool for learning from
  past successes/failures, Tree-of-Thought strategy, multi-environment
  support (software, games, mobile). Software company metaphor with
  ProductManager → Architect → Engineer → QAEngineer pipeline.
- [SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory) — upstream
- [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) — SWE coding agent
- [allenai/SAGE](https://github.com/allenai/SAGE) — Gemini with caching
- [ExtensityAI/symbolicai](https://github.com/ExtensityAI/symbolicai) — Gemini reasoning

## Architecture Comparison

### Agent Systems

| Feature | AgentLaboratory | AI-Researcher | AutoAgent | DeepCode | MetaGPT |
|---------|----------------|---------------|-----------|----------|---------|
| **Agent roles** | PhD, MLE, SWE, Postdoc, Prof, Reviewers | Survey, Idea, Plan, ML, Prepare, ExpAnalyser, Judge | Meta-agents that CREATE agents dynamically | Orchestration engine with plugin agents | ProductManager, Architect, Engineer, QAEngineer, Researcher, etc. |
| **Code execution** | Local multiprocessing (600s timeout) | Docker sandbox via TCP | Docker sandbox via TCP | Docker + MCP tool servers (nanobot) | Local with structured code review |
| **GPU** | Local (+ Kaggle) | Docker GPU passthrough | Docker GPU passthrough | Docker GPU + cloud | Not focused on GPU |
| **Workflow** | Linear phases | FlowGraph DAG + FlowCache | Event-driven flow engine (async) | Agent orchestration engine (78KB) | Environment pub/sub with n_round loop |
| **Self-review** | ReviewersAgent on final report | JudgeAgent at each iteration | Self-correcting via meta-agents | Multi-stage verification | QAEngineer writes and runs tests |
| **Agent creation** | Static (hardcoded) | Static (predefined) | **Dynamic** (LLM creates agents) | Plugin-based | Static roles with dynamic actions |
| **Tool creation** | Static | Static | **Dynamic** (LLM creates tools) | MCP tool servers | Static with extensible actions |
| **Memory** | List-based history | FlowCache | Registry + persistent memory | Workspace-based | **Hierarchical**: short-term, long-term, brain, RoleZero |
| **Experience learning** | None | None | None | None | **Experience Pool** with scoring and retrieval |
| **Communication** | Direct function calls | Direct function calls | Direct function calls | API-based | **Message bus** (pub/sub Environment) |
| **Budget control** | None | None | None | None | **CostManager** with NoMoneyException |
| **Strategy** | None | None | None | None | **ToT, Planner, ExperienceRetriever** |
| **LLM backend** | OpenAI, Gemini, DeepSeek, Anthropic | LiteLLM | LiteLLM | Multi-provider | Multi-provider via config |
| **Serialization** | Pickle state save | FlowCache | Registry | Workspace | **Pydantic-based** serialize/deserialize |

### Key Insights from Each Project

#### AI-Researcher: Docker Sandbox + Quality Gates
- Runs ALL code in Docker containers via TCP server
- `JudgeAgent` evaluates experiments at each iteration
- `ExpAnalyser` provides structured result analysis
- FlowGraph DAG allows non-linear workflow execution
- FlowCache prevents redundant LLM calls

#### AutoAgent: Meta-Agent Pattern (Most Novel)
- **AgentCreator**: LLM dynamically creates new agent types at runtime
- **ToolEditor**: LLM creates new tools based on task requirements
- **WorkflowCreator/Former**: LLM designs multi-agent workflows
- **EventEngine**: Async event-driven flow with group triggers
- **Registry**: Runtime agent/tool/workflow registration and discovery
- This is the most advanced pattern — agents that design themselves

#### DeepCode: Paper-to-Code Pipeline
- Transforms research papers into working code implementations
- 78KB `agent_orchestration_engine.py` — massive orchestration system
- MCP (Model Context Protocol) tool servers via nanobot
- Docker-based sandboxed execution environment
- FastAPI backend + React frontend for modern UI
- Plugin system for extensible agent capabilities

#### Auto-Deep-Research: Deep Research Pipeline
- File selection agent for relevant document discovery
- Web search integration for literature
- Built on the AutoAgent framework (shared codebase)
- Focused on deep, multi-step research tasks

#### DeepResearch-Eval: Research Quality Evaluation
- `judge_fact.py`: Fact-checking with structured prompts
- `judge_score.py`: Multi-dimension scoring (depth, breadth, accuracy)
- `Aprompts.py`: 17KB of structured evaluation prompts
- `Atools.py`: Evaluation tooling

#### OpenHands / SWE-bench: Live Coding Agents
- Agents that can edit files, run commands, browse web
- Docker sandbox with full development environment
- Self-planning and re-planning based on execution results
- Continuous iteration until task is solved

#### MetaGPT: Structured Multi-Agent Framework (Most Mature)
- **Role-Action-Message pattern**: Each Role observes Messages,
  selects an Action, executes it, publishes results as new Messages.
  This decouples agents from each other — they communicate ONLY via
  the shared Environment message bus, never by direct function calls.
- **Environment as shared memory**: The Environment holds the message
  bus, role registry, and shared context. Agents publish/subscribe to
  messages by type. This is fundamentally different from AgentLaboratory
  where agents call each other directly.
- **Team with budget control**: `Team` wraps the Environment and adds
  investment/budget tracking. If costs exceed the budget, it raises
  `NoMoneyException`. This prevents runaway API costs.
- **Experience Pool (`exp_pool`)**: Records past task attempts with
  scoring. The `ExperienceRetriever` finds similar past experiences to
  guide current tasks. This is the most advanced learning-from-experience
  system among all analyzed projects.
- **Hierarchical memory**: `Memory` (short-term working memory),
  `LongTermMemory` (persistent across sessions), `BrainMemory` (with
  summarization), `RoleZeroMemory` (for the base role). AgentLaboratory
  only has a flat list.
- **Strategy module**: Tree of Thought (`tot.py`), Planner, Solver,
  ThinkingCommand, SearchSpace. These provide structured reasoning
  strategies beyond simple chain-of-thought prompting.
- **Structured schema** (`schema.py`, 33KB): Pydantic models for
  Messages, Documents, CodePlanAndChange, etc. All inter-agent
  communication uses typed, serializable objects.
- **Serialization**: Full serialize/deserialize for Teams and Roles
  via Pydantic, enabling state save/restore across sessions.
- **QAEngineer role**: Dedicated role that writes and runs tests for
  generated code. AgentLaboratory has no automated testing of generated
  code.

## What AgentLaboratory Already Has (Advantages)

1. **Kaggle integration**: Remote GPU via Kaggle notebooks with log monitoring
2. **Sub-agent pipeline**: 8 specialized agents for training pipeline
3. **Tavily-powered research**: Dynamic web search for best practices
4. **Gemini 3 Pro + Flash**: Extended thinking (24576 budget) + cheap monitoring
5. **Multi-LLM without LiteLLM**: Direct provider integration
6. **CPU/GPU testing pipeline**: CPU-first validation before GPU training

## Improvements Already Implemented in This PR

- `.env.example` — environment variable templates for all API keys
- `Dockerfile` + `docker-compose.yml` — Kaggle Docker image, CPU/GPU profiles
- `kaggle_utils.py` — dataset listing, notebook submission, log monitoring
- `kaggle_training_pipeline.py` — self-iterating CPU→GPU pipeline
- `pipeline_subagents.py` — 8 specialized sub-agents with Tavily research
- `inference.py` — Gemini 3 Pro/Flash with max thinking budget
- `tools.py` — `execute_code_kaggle()`, `detect_device()`, execution logging
- `mlesolver.py` — Kaggle routing, error history tracking
- `ai_lab_repo.py` — Kaggle config passthrough, error/execution logs

## Actionable Improvements (Prioritized)

### Priority 1: Message Bus Architecture (from MetaGPT)

**Problem**: AgentLaboratory agents communicate via direct function calls
and pass state through method parameters. This creates tight coupling —
adding or removing an agent requires modifying the workflow code.

**Why it matters**: MetaGPT's pub/sub Environment lets agents communicate
via typed Messages without knowing about each other. This enables:
adding new agents without changing existing code, parallel agent execution,
and replay/debugging of agent communication.

**Implementation**: Create a simple `MessageBus` class. Each agent
publishes Messages and subscribes to message types. The workflow loop
in `ai_lab_repo.py` drives the bus rather than calling agents directly.

### Priority 2: Experience Pool (from MetaGPT)

**Problem**: AgentLaboratory doesn't learn from past experiments. Each
run starts from scratch with no memory of what worked or failed before.

**Why it matters**: MetaGPT's `exp_pool` records past attempts with
scoring, then retrieves similar experiences to guide new tasks. This
dramatically improves success rates over iterations.

**Implementation**: Store experiment results (code, output, errors, scores)
in a JSON-based experience pool. Before generating code, retrieve similar
past experiences and include them as few-shot examples in the prompt.

### Priority 3: Budget/Cost Control (from MetaGPT)

**Problem**: AgentLaboratory has no cost controls. A runaway experiment
can consume unlimited API tokens.

**Why it matters**: MetaGPT's `CostManager` with `NoMoneyException`
provides a safety net. Team.invest() sets a budget, and the system
stops when costs exceed it.

**Implementation**: Add token counting to `inference.py` for all
providers (OpenAI already tracked, add Gemini via `usage_metadata`).
Add a `max_budget` parameter to `LaboratoryWorkflow`. Raise an
exception when the budget is exhausted.

### Priority 4: Meta-Agent Pattern (from AutoAgent)

**Problem**: AgentLaboratory has hardcoded agents. AutoAgent's meta-agents
dynamically create new agents, tools, and workflows based on the task.

**Why it matters**: Different research tasks need different agent
configurations. A GNN paper needs different tools than a diffusion paper.

**Implementation**: Add a `MetaAgent` that analyzes the research topic and
dynamically configures the agent pipeline — selecting which agents to
activate, what tools to provide, and how to structure the workflow. Start
simple: let the MetaAgent choose between existing agents and configure
their parameters rather than generating entirely new agents from scratch.

### Priority 5: Event-Driven Flow Engine (from AutoAgent)

**Problem**: AgentLaboratory's linear phase loop in `ai_lab_repo.py` is
rigid. AutoAgent's `EventEngine` uses async event-driven execution with
group triggers, allowing parallel and conditional workflows.

**Why it matters**: Some phases can run in parallel (e.g., literature
review and data preparation). Failed experiments should trigger
re-planning without restarting everything.

**Implementation**: Replace the `for subtask in subtasks` loop with an
event-driven system where phases emit completion events that trigger
dependent phases. Use Python's `asyncio` for concurrent execution.

### Priority 6: Docker-Sandboxed Execution (from AI-Researcher + DeepCode)

**Problem**: AgentLaboratory runs generated code directly on the host.

**Why it matters**: Generated ML code can corrupt the environment, consume
all memory, or run indefinitely. Both AI-Researcher and DeepCode sandbox
execution in Docker containers.

**Implementation**: Route `execute_code()` through Docker when available,
using the existing Dockerfile. Connect via TCP (AI-Researcher pattern) or
MCP tool servers (DeepCode/nanobot pattern).

### Priority 7: Structured Evaluation (from DeepResearch-Eval)

**Problem**: AgentLaboratory's ReviewersAgent uses freeform text evaluation.
DeepResearch-Eval uses structured multi-dimension scoring with fact-checking.

**Why it matters**: Structured evaluation catches specific failure modes
(factual errors, missing baselines, weak methodology) that freeform
review misses.

**Implementation**: Add structured evaluation prompts (inspired by
DeepResearch-Eval's `Aprompts.py`) that score on: factual accuracy,
experimental rigor, novelty, reproducibility, and methodology soundness.
Add `judge_fact`-style fact-checking for key claims.

### Priority 8: JudgeAgent Quality Gates (from AI-Researcher)

**Problem**: AgentLaboratory only evaluates the final report.

**Implementation**: Add a JudgeAgent that runs after each experiment phase,
scoring results and deciding whether to iterate or proceed. Uses structured
scoring criteria rather than freeform review.

### Priority 9: Dual Model Strategy (from AI-Researcher)

**Problem**: Same expensive model for all tasks.

**Implementation**: Use Gemini 3 Flash for: log monitoring, simple
formatting, status checks, and routine tasks. Reserve Gemini 3 Pro for:
code generation, complex reasoning, and review.

### Priority 10: Response Caching (from AI-Researcher FlowCache)

**Problem**: Repeated identical LLM calls waste tokens and money.

**Implementation**: Hash (model + system_prompt + prompt) and cache
responses. Use file-based cache with configurable TTL.

### Priority 11: MCP Tool Servers (from DeepCode/nanobot)

**Problem**: Tools are hardcoded Python functions.

**Why it matters**: DeepCode's nanobot uses MCP (Model Context Protocol)
tool servers, enabling tools to run in separate processes with proper
isolation, and allowing dynamic tool discovery.

**Implementation**: Wrap existing tools (code execution, file I/O, web
search) as MCP tool servers. This enables: tool isolation, remote tools,
and dynamic tool registration.

### Priority 12: Agent/Tool Registry (from AutoAgent)

**Problem**: Agents and tools are imported directly, no runtime discovery.

**Implementation**: Add a registry system where agents, tools, and
workflows register themselves. The MetaAgent can then discover and compose
available capabilities at runtime.

### Priority 13: Gemini Cost Tracking

**Problem**: Cost tracking exists for OpenAI but not Gemini.

**Implementation**: Track `response.usage_metadata` for Gemini API calls
and calculate costs based on current pricing.

### Priority 14: Hierarchical Memory (from MetaGPT)

**Problem**: AgentLaboratory stores conversation history as a flat list
with a max length of 15. There's no long-term memory, no summarization
of old context, and no persistent memory across sessions.

**Why it matters**: MetaGPT has 5 memory types: Memory (working),
LongTermMemory (persistent), BrainMemory (with summarization),
RoleZeroMemory (foundational), and MemoryStorage (file-backed).

**Implementation**: Add a `BrainMemory`-like summarization layer that
compresses old conversation history into summaries when the context
window fills up. Add persistent memory storage so experiments can be
resumed across sessions with full context.

### Priority 15: QA/Testing Agent (from MetaGPT)

**Problem**: AgentLaboratory generates code but never tests it before
running. If the code has syntax errors or import issues, the entire
600-second execution timeout is wasted.

**Why it matters**: MetaGPT's QAEngineer role writes and runs tests
for generated code before deployment. This catches errors early.

**Implementation**: Add a lightweight `QAAgent` that runs basic
validation on generated code before execution: syntax check, import
check, type annotation check. For ML code, validate that the training
loop structure is correct (has loss.backward(), optimizer.step(), etc.).

### Priority 16: Structured Schema (from MetaGPT)

**Problem**: AgentLaboratory passes data between agents as unstructured
strings. Parsing relies on regex extraction of JSON/code blocks, which
is fragile and error-prone.

**Why it matters**: MetaGPT uses Pydantic-based `Message` and
`Document` types (33KB schema.py) for all inter-agent communication.
This makes data exchange reliable and debuggable.

**Implementation**: Define Pydantic models for key data types:
`ExperimentPlan`, `CodeBlock`, `ReviewFeedback`, `ExperimentResult`.
Use structured output / JSON mode where supported by the LLM provider.
