# Kaggle Best Practices for ML Training Pipelines

Best practices for logging, monitoring, GPU acceleration, and code quality
when running ML training on Kaggle notebooks. Derived from Kaggle documentation
and patterns from high-performing Kaggle projects.

## 1. Logging and Monitoring

### Structured Logging
- Use `print(..., flush=True)` for all output — Kaggle buffers stdout
- Prefix log lines with structured tags: `[INFO]`, `[ERROR]`, `[METRIC]`
- Include elapsed time in every log line for debugging timeouts
- Log system info at startup: Python version, PyTorch version, CUDA availability

### Training Progress
- Log loss and metrics every N batches (not every batch — reduces overhead)
- Log epoch-level summaries with: epoch, train_loss, val_loss, accuracy, time
- Use format: `[METRIC epoch=1 train_loss=0.45 val_loss=0.52 acc=0.78 time=45s]`
- Log learning rate changes and optimizer state

### Error Handling
- Wrap entire training in try/except with full traceback logging
- Log OOM errors with current batch size and model size
- Save partial checkpoints before risky operations (large batch, new epoch)
- Catch and log `torch.cuda.OutOfMemoryError` specifically

### Resource Monitoring
- Log GPU memory usage: `torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()`
- Log CPU memory: `psutil.virtual_memory().percent` (if psutil available)
- Log disk usage before saving checkpoints
- Monitor training speed: samples/second, batches/second

## 2. GPU Acceleration (Kaggle-Specific)

### Device Setup
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)
```

### Kaggle GPU Limits
- **GPU quota**: ~30 hours/week for P100, ~20 hours/week for T4
- **Session limit**: 9 hours max per GPU session, 12 hours per CPU session
- **Memory**: P100 has 16GB VRAM, T4 has 16GB VRAM
- **Internet**: Available during setup, may be disabled during competition runs

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### DataLoader Optimization
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,       # Kaggle has 2-4 CPU cores
    pin_memory=True,     # Faster GPU transfer
    prefetch_factor=2,   # Prefetch batches
    persistent_workers=True,
)
```

### Memory Management
- Clear cache between epochs: `torch.cuda.empty_cache()`
- Use gradient accumulation for large effective batch sizes
- Delete intermediate tensors explicitly: `del tensor; torch.cuda.empty_cache()`
- Use `torch.no_grad()` for validation/inference

## 3. Code Quality for Kaggle

### Self-Contained Scripts
- Import all dependencies at top of file
- No `!pip install` or cell magic — use script format
- Handle missing packages gracefully with try/except ImportError
- Set random seeds for reproducibility

### Checkpoint Management
- Save model checkpoints every N epochs
- Save to current directory (Kaggle output)
- Include epoch number and metric in checkpoint filename
- Save optimizer state for resume capability

### Kaggle Environment
- Pre-installed packages: torch, tensorflow, sklearn, pandas, numpy, transformers
- Working directory: `/kaggle/working/`
- Input data: `/kaggle/input/`
- Output saved from: `/kaggle/working/` (current directory)

## 4. Monitoring Sub-Agent Best Practices

### For the Monitoring Agent (Gemini 2.5 Flash)
- Use lightweight model (Flash) for cost efficiency during monitoring
- Check logs every 20-30 seconds for real-time feedback
- Parse structured log tags to extract metrics automatically
- Detect error patterns: OOM, NaN loss, divergence, import errors
- Early termination signals:
  - Loss becomes NaN or Inf
  - No progress for 5+ minutes
  - Memory usage > 90%
  - Same error repeated 3+ times

### For the Code Review Agent (Gemini 2.5 Pro)
- Use high-thinking model for thorough code analysis
- Check for: device mismatches, missing imports, data leaks, memory leaks
- Verify Kaggle constraints: file paths, internet access, time limits
- Ensure logging code is present and properly structured

## 5. Iterative Improvement Patterns

### Error Learning
- Accumulate errors across iterations
- Pass error history to code generation prompts
- Classify errors: syntax, runtime, resource, convergence
- Apply targeted fixes based on error category

### Metrics-Driven Iteration
- Compare metrics across iterations
- If accuracy plateaus: suggest architecture changes
- If training is slow: suggest optimization changes
- If OOM: suggest batch size / model size reduction
