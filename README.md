# IndicTrans2 LoRA for Conversational English–Hindi Translation

A parameter-efficient fine-tuning project using **IndicTrans2-200M** with a saved **LoRA adapter** for English-to-Hindi translation. The included evaluation workflow compares the LoRA-adapted model against the original base model on a 1,000-sentence validation set.

## Overview

This project evaluates a saved LoRA-fine-tuned checkpoint built on `ai4bharat/indictrans2-en-indic-dist-200M` for English-to-Hindi translation.

The evaluation compares:
- The original IndicTrans2 base model
- The saved LoRA checkpoint (`checkpoint-6250`)
- A 1,000-example validation set

Both models are evaluated with the same preprocessing and decoding configuration using BLEU and chrF++.

## Results

### Full Validation Set — 1,000 Sentences

| Metric | Base Model | LoRA Model | Improvement |
|---|---:|---:|---:|
| BLEU | 13.30 | **16.61** | **+3.31** |
| chrF++ | 33.49 | **36.97** | **+3.48** |

Evaluation used **5-beam decoding**.

### Quick 200-Sentence Evaluation

| Metric | Base Model | LoRA Model | Improvement |
|---|---:|---:|---:|
| BLEU | 13.06 | **17.58** | **+4.52** |
| chrF++ | 34.42 | **38.35** | **+3.93** |

## Model & Configuration

| Setting | Value |
|---|---|
| Base model | `ai4bharat/indictrans2-en-indic-dist-200M` |
| Source language | `eng_Latn` |
| Target language | `hin_Deva` |
| Maximum token length | 128 |
| LoRA checkpoint | `checkpoint-6250` |
| Validation examples | 1,000 |
| Beam size | 5 |
| Random seed | 42 |

The LoRA adapter is loaded using **PEFT** on top of a fresh copy of the IndicTrans2 base model.

## Evaluation Workflow

```text
English Input
     │
     ▼
IndicProcessor Preprocessing
     │
     ├───────────────┐
     ▼               ▼
Base IndicTrans2   IndicTrans2 + LoRA
     │               │
     ▼               ▼
Hindi Prediction   Hindi Prediction
     │               │
     └───────┬───────┘
             ▼
       Reference Hindi
             │
        ┌────┴────┐
        ▼         ▼
      BLEU      chrF++
```

The notebook performs:
1. Environment setup
2. Model and LoRA checkpoint loading
3. Single-sentence sanity check
4. 200-sentence quick evaluation
5. Full 1,000-sentence evaluation
6. BLEU and chrF++ comparison

## Repository Contents

The provided project contains the evaluation notebook:

```text
IndicTrans2_LoRA_Evaluation_CLEAN.ipynb
```

The notebook is specifically designed for clean evaluation of the saved LoRA checkpoint and does not contain the original training loop.

## Requirements

Main dependencies used by the notebook:

```text
transformers==4.45.2
IndicTransToolkit
accelerate
peft
sacrebleu
sentencepiece
datasets
indic-nlp-library
sacremoses
torchao>0.16.0
```

A CUDA-enabled runtime is recommended. The recorded evaluation used an **NVIDIA Tesla T4** GPU.

## Running the Evaluation

The notebook is designed for **Google Colab**.

1. Start a fresh/restarted Colab runtime.
2. Install the dependencies from the setup cells.
3. Mount Google Drive.
4. Make the LoRA checkpoint and validation set available at the configured paths.
5. Run the sanity check.
6. Run the 200-sentence evaluation.
7. Run the complete validation evaluation.

Expected project paths in the notebook:

```text
NMT_Project_Conv/
├── indictrans2_lora_conv/
│   └── checkpoint-6250/
└── val_raw_conv/
```

## Reproducibility

The evaluation configuration includes:

- Random seed: `42`
- GPU inference with `float16`
- Maximum token length: `128`
- Batch size: `32` on CUDA
- Beam size: `5`
- `no_repeat_ngram_size=3`
- Early stopping enabled
- `use_cache=True` during generation

## Hugging Face Authentication

The base model is public, so authentication is normally optional. If authentication is required for a private or gated resource, use **Google Colab Secrets** rather than hardcoding credentials.

**Never commit Hugging Face tokens or other secrets to the repository.**

## Limitations

- The available notebook focuses on evaluating the saved LoRA checkpoint rather than reproducing the original training run.
- Evaluation is based on a 1,000-sentence validation set.
- Results depend on the checkpoint, decoding configuration, preprocessing, and validation data.
- Automatic evaluation currently uses BLEU and chrF++.

## Future Improvements

- Evaluate on larger and more diverse conversational test sets.
- Add human evaluation for fluency and conversational naturalness.
- Compare additional decoding configurations.
- Measure inference efficiency alongside translation quality.
- Extend evaluation to additional Indic languages.

## Conclusion

The evaluated LoRA checkpoint improves English-to-Hindi translation quality over the base IndicTrans2 model:

**BLEU:** 13.30 → **16.61** (+3.31)  
**chrF++:** 33.49 → **36.97** (+3.48)

These results show a measurable improvement from the LoRA-adapted model on the evaluated validation set.

## Repository

**GitHub:** 
https://github.com/sahithipriya426/IndicTrans2-LoRA

