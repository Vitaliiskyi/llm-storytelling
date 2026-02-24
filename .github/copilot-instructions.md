# AI Coding Agent Instructions

## Project Overview
This is an **LLM fine-tuning project** for training language models on folk tale data. The pipeline follows a strict data→tokenize→train→evaluate workflow:
1. **Data Source**: CSV file (`folk_tales_deduplicated.csv`) with folk tale stories
2. **Preprocessing**: Convert CSV to JSONL format (`data_prep.py`)
3. **Training**: Fine-tune Pythia models using HuggingFace Transformers (`train.py`)
4. **Inference**: Test trained models with prompts (`test_new_model.py`)

## Architecture Patterns

### Data Pipeline (CSV → JSONL → Training)
- **Input**: `folk_tales_deduplicated.csv` with columns `[source, nation, title, text]`
- **Processing** (`data_prep.py`): Converts to JSONL with `{"title": "...", "completion": "..."}` format
- **Key Detail**: Prepends space to completion text to match tokenization expectations
- **Output**: `stories_10.jsonl` with 2939+ examples

### Tokenization Pattern (Critical)
See `train.py` lines 45-88:
- **Title-as-context**: Titles are tokenized WITHOUT special tokens, prepended to each completion
- **Chunking with stride**: Long completions are split into overlapping chunks (stride=256) to fit MAX_LENGTH=2048
- **Label generation**: Labels mirror input_ids for causal language modeling (MLM=False)
- **Return format**: Dict with `input_ids`, `attention_mask`, `labels` (each is list of sequences)
- **Critical Flag**: `batched=False` in map() - must remain False for list-returning tokenize function

### Training Configuration (Pythia-410M)
- **Model**: `EleutherAI/pythia-410m` (410 million parameters)
- **Hardware Optimization** (`train.py` lines 95-115):
  - `fp16=True` for mixed precision (RTX 3090 memory constraint)
  - `per_device_train_batch_size=2` (small due to VRAM)
  - `gradient_accumulation_steps=2` (simulates batch size 4)
  - `dataloader_num_workers=4` + `pin_memory=True` (data loading parallelization)
- **Checkpoints**: Saves every 500 steps, keeps only 3 latest
- **Output Structure**: `./result_model/pythia-410-finetuned-exp-{N}epochs{TIMESTAMP}/`

### Inference Pattern
- Load checkpoint from `result_model/` subdirectory
- Use HuggingFace `pipeline("text-generation")` with model and tokenizer
- Add pad token to tokenizer before inference: `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`
- Prompt format: `"Title: {title}\n\nStory:\n"` (matches training format)

## Developer Workflows

### Run Full Training Pipeline
```bash
python data_prep.py          # CSV → JSONL (updates stories_10.jsonl)
python train.py              # Fine-tune and save to result_model/
# Check ./logs for training metrics
```

### Debug Training Issues
- **Check last error**: Look at `Train.py` error message - often tokenization or memory related
- **GPU issues**: Run `gpu_check.py` to verify CUDA availability
- **Data format**: Ensure `stories_10.jsonl` exists and has proper `{"title", "completion"}` schema
- **Token limit**: If OOM errors, reduce `per_device_train_batch_size` or `MAX_LENGTH`

### Test New Model
```bash
python test_new_model.py     # Load checkpoint and generate text
# Adjust checkpoint path: pythia-410-finetuned-exp-4epochs2025-12-08.../checkpoint-5876/
```

### Inference on New Data
Models expect title-only prompts. Modify prompt in test script:
```python
prompt = "Title: Dragon and The boy\n\nStory:\n"
```

## Critical Conventions

### File Organization
- Raw data: `folk_tales_deduplicated.csv` (source of truth)
- Processing scripts: `data_prep.py`, `utils.py` (currently empty)
- Training: `train.py` (main entry point)
- Model testing: `test_new_model.py`, `qwen3.py` (experimental backends)
- Models: `./mistral-7b/` (reference), `./result_model/` (outputs)

### Tokenization Gotchas
1. **Always add pad token**: `tokenizer.add_special_tokens({'pad_token': '[PAD]'})` before training/inference
2. **Title-completion structure**: Training expects title prepended to completion in tokenizer
3. **Chunking return format**: tokenize() must return dict of lists (one per chunk), not single sequences
4. **no_input_ids in labels**: Labels are copy of input_ids for causal LM loss calculation

### Model Checkpoints
- Located in `result_model/pythia-410-finetuned-exp-{N}epochs{TIMESTAMP}/checkpoint-{STEP}/`
- Each has: `config.json`, `generation_config.json`, `model.safetensors` (or `.bin`)
- To use: Pass checkpoint path to `AutoModelForCausalLM.from_pretrained()`

### Data Constraints
- CSV has 3206 rows, stories_10.jsonl has 2939 JSONL lines (after filtering empty/null)
- Average story ~500-1000 tokens
- Stories are international folk tales (Japanese, Serbian, German, French origins)

## Testing & Validation
- No unit tests currently exist; focus on integration testing
- Verify training output: check `./logs/` for loss curves, `result_model/` for saved checkpoints
- Validate inference: prompt model with story title and verify coherent continuation
- Memory baseline: RTX 3090 with current settings = ~8-10 GB VRAM usage

## When to Call Out
- Models fail to load (safetensors vs .bin format)
- CUDA out-of-memory errors (reduce batch size or sequence length)
- Data pipeline breaks (check CSV encoding, JSONL formatting)
- Tokenizer pad token missing (add before any forward pass)
