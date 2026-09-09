# IndicTrans2 LoRA for Conversational English–Hindi Translation

Parameter-efficient fine-tuning of **AI4Bharat IndicTrans2-200M** for conversational and colloquial English–Hindi translation using **LoRA**, with a separate evaluation workflow comparing the base model against the fine-tuned adapter.

## 1. Project Overview

This project adapts `ai4bharat/indictrans2-en-indic-dist-200M` to conversational English–Hindi text. The training data is sourced from the **OpenSubtitles** English–Hindi parallel corpus, which contains dialogue-heavy and informal language such as contractions, short utterances, slang, and conversational phrasing.

The project is organized around two notebooks:

- **Training notebook** — data loading, cleaning, preprocessing, baseline evaluation, LoRA fine-tuning, checkpoint selection, and an initial comparison.
- **Clean evaluation notebook** — independently reloads the base model and saved LoRA checkpoint, then evaluates both models on the held-out validation set.

The separation of training and evaluation makes the final comparison reproducible and avoids relying on model state left in memory after training.

---

## 2. Objectives

- Adapt IndicTrans2 to conversational/colloquial English–Hindi translation.
- Use **LoRA** instead of full-model fine-tuning to reduce the number of trainable parameters.
- Establish a zero-shot baseline before adaptation.
- Compare the base and LoRA models using **BLEU** and **chrF++**.
- Validate the improvement on a held-out set of **1,000 sentence pairs**.
- Perform qualitative checks on conversational, slang-heavy, and informal inputs.
- Preserve correct IndicTrans2 target preprocessing to avoid label corruption.

---

## 3. Base Model

**Model:** `ai4bharat/indictrans2-en-indic-dist-200M`

- Source language: `eng_Latn`
- Target language: `hin_Deva`
- Maximum token length: 128
- Generation: beam search with `num_beams=5` for the GPU evaluation run
- Random seed: 42

IndicTrans2 is used as the pretrained sequence-to-sequence translation model, while LoRA learns a small set of task/domain-specific adapter parameters.

---

## 4. Dataset and Data Preparation

### Conversational Parallel Corpus

The training notebook loads English–Hindi parallel data from **OpenSubtitles** using the Hugging Face `datasets` library. A fallback OPUS-100 loader is included if the OpenSubtitles loader is unavailable.

The preprocessing pipeline performs:

1. Extraction of English/Hindi translation pairs.
2. Removal of empty pairs.
3. Exact-pair deduplication.
4. Minimum and maximum length filtering.
5. English/Hindi length-ratio filtering.
6. Devanagari-character validation on the Hindi side.
7. Shuffling with a fixed random seed.
8. Separation into training and validation subsets.

The configured target sizes are:

- Training: up to **50,000** cleaned examples
- Validation: **1,000** examples

In the recorded training notebook run, the resulting split contained **488 training pairs and 1,000 validation pairs** after the available corpus/filtering constraints.

> The notebook's code is parameterized for up to 50,000 training examples; the recorded run should be described using the actual resulting split rather than the configured maximum.

---

## 5. IndicTrans2 Preprocessing

The project uses `IndicProcessor` before tokenization.

A critical implementation detail is the different treatment of source inputs and target labels:

- **Source / encoder text:** `is_target=False`
- **Target / decoder labels:** `is_target=True`

For the target side, `is_target=True` prevents the target language tag from being incorrectly inserted into the decoder labels.

The resulting tokenized datasets are cached to Google Drive:

```text
tokenized_train_conv/
tokenized_val_conv/
```

A two-example sanity check is performed before preprocessing the complete datasets to verify that Hindi labels do not begin with an unintended language-tag sequence.

---

## 6. LoRA Fine-Tuning

The model is adapted using **PEFT-LoRA** with the following configuration:

| Parameter | Value |
|---|---:|
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Bias | `none` |
| Task type | `SEQ_2_SEQ_LM` |
| Learning rate | `1e-4` |
| Scheduler | Linear |
| Warmup ratio | 0.10 |
| Weight decay | 0.0 |
| Max gradient norm | 1.0 |
| Epochs | 1 |
| Train batch size/device | 2 |
| Eval batch size/device | 2 |
| Gradient accumulation | 4 |
| Evaluation interval | Every 500 steps |
| Checkpoint interval | Every 500 steps |
| Best-model criterion | Lowest evaluation loss |

The effective batch size from the configured training batch and gradient accumulation is 8 examples per optimizer update.

Training uses FP16 when CUDA is available.

The best checkpoint selected during training was:

```text
checkpoint-6250
```

---

## 7. Evaluation Methodology

The evaluation notebook is intentionally separate from training.

For evaluation:

1. Start from a fresh runtime.
2. Load the pretrained IndicTrans2 base model.
3. Load the saved LoRA adapter from `checkpoint-6250`.
4. Generate translations for the same held-out validation set.
5. Compute BLEU and chrF++ independently for the base and LoRA models.
6. Compare the resulting scores.
7. Inspect qualitative translations.

The full evaluation contains **1,000 validation sentences** and uses **5-beam decoding**.

### Metrics

**BLEU** measures n-gram overlap between generated translations and reference translations.

**chrF++** evaluates character n-gram similarity with word-order information and is particularly useful for morphologically rich languages such as Hindi.

---

## 8. Results

### Full Validation Set — 1,000 Sentences

| Metric | Base IndicTrans2 | LoRA Fine-Tuned | Improvement |
|---|---:|---:|---:|
| BLEU | 13.30 | **16.61** | **+3.31** |
| chrF++ | 33.49 | **36.97** | **+3.48** |

Exact recorded values:

- Base BLEU: `13.299960315075303`
- LoRA BLEU: `16.60880087296448`
- Base chrF++: `33.494087926881974`
- LoRA chrF++: `36.97433009327248`

The LoRA-adapted model therefore improves both automatic metrics on the held-out conversational validation set.

### Quick Evaluation

A smaller 200-sentence evaluation was also recorded:

| Metric | Base | LoRA | Improvement |
|---|---:|---:|---:|
| BLEU | 13.06 | 17.58 | +4.52 |
| chrF++ | 34.42 | 38.35 | +3.93 |

The 1,000-sentence evaluation should be treated as the primary reported result because it provides the larger evaluation sample.

---

## 9. Qualitative Evaluation

The evaluation notebook includes qualitative spot checks comparing:

```text
English input
Reference Hindi
Base-model translation
LoRA translation
```

Additional conversational/adversarial probes target:

- slang and informal vocabulary
- contractions
- idiomatic expressions
- conversational address
- colloquial phrases
- short dialogue-style sentences

Examples include:

```text
What's up, dude? Long time no see.
I'm gonna grab some grub, you in?
My bad, I didn't mean to mess it up.
Get out of here! You're kidding me.
Chill out, it's not a big deal.
```

These examples are useful for examining whether domain adaptation changes conversational behavior beyond what aggregate BLEU/chrF++ scores capture.

The qualitative outputs show that the LoRA model can still produce awkward translations on some slang-heavy inputs, demonstrating that the improvement in aggregate metrics does not mean conversational translation is completely solved.

---

## 10. Evaluation Reproducibility

The clean evaluation notebook reloads:

```text
Base model:
ai4bharat/indictrans2-en-indic-dist-200M

LoRA checkpoint:
checkpoint-6250
```

The evaluation uses:

```text
Validation examples: 1000
num_beams: 5
Source: eng_Latn
Target: hin_Deva
```

The evaluation notebook also clears stray Hugging Face environment tokens and recommends using Colab Secrets rather than hardcoding credentials when authentication is required.

**Never commit real Hugging Face tokens or other credentials to the repository.**

---

## 11. Project Structure

A recommended repository structure is:

```text
.
├── README.md
├── training/
│   └── indictrans2_lora_conversational_training.ipynb
├── evaluation/
│   └── IndicTrans2_LoRA_Evaluation_CLEAN.ipynb
├── reports/
│   └── IndicTrans2_LoRA_Project_Report.tex
└── .gitignore
```

Large generated artifacts such as model checkpoints and cached datasets should generally remain outside Git unless there is a specific reason to version them.

---

## 12. Requirements

The notebooks use:

```text
transformers
IndicTransToolkit
accelerate
peft
sacrebleu
sentencepiece
datasets
indic-nlp-library
sacremoses
torch
pandas
numpy
tqdm
huggingface_hub
```

The training notebook pins:

```text
transformers==4.28.0
```

for compatibility with the IndicTransToolkit setup used there.

The clean evaluation notebook pins:

```text
transformers==4.45.2
```

for its evaluation environment.

Because the notebooks use different pinned Transformers versions, it is preferable to run them in their intended notebook/runtime environments rather than assuming one environment will reproduce both workflows unchanged.

---

## 13. Running the Project

### Training

Open the training notebook in Google Colab and run the cells in order.

The notebook:

1. Mounts Google Drive.
2. Installs dependencies.
3. Loads OpenSubtitles English–Hindi data.
4. Cleans and filters the corpus.
5. Creates train/validation splits.
6. Preprocesses and tokenizes the data.
7. Evaluates the base model.
8. Applies LoRA.
9. Fine-tunes with `Seq2SeqTrainer`.
10. Saves checkpoints and training metadata.
11. Evaluates the LoRA model.

### Evaluation

Use the clean evaluation notebook separately.

Before running it:

1. Restart the Colab runtime.
2. Mount the Drive containing the saved adapter.
3. Confirm the `checkpoint-6250` path.
4. Load the base model.
5. Attach the LoRA adapter.
6. Run the 1,000-sentence evaluation.
7. Inspect the quantitative and qualitative results.

---

## 14. Important Implementation Lesson

One of the most important engineering details in this project was correct target preprocessing.

For IndicTrans2:

```python
target_texts = ip.preprocess_batch(
    hindi,
    src_lang=tgt_lang,
    tgt_lang=src_lang,
    is_target=True
)
```

Using the default source-style preprocessing for decoder labels can introduce an unintended language tag into every target sequence. The notebook explicitly guards against this by using `is_target=True` and performs a small label sanity check before processing the complete dataset.

This is an important reproducibility detail because preprocessing bugs can appear as apparent model-training failures when the underlying issue is corrupted target construction.

---

## 15. Limitations

- The main evaluation is based on a held-out OpenSubtitles-derived split rather than the official IN22-Conv benchmark.
- The recorded training split contains only 488 examples despite a configurable target of 50,000, so the reported run should not be presented as a 50k-example training run.
- Conversational translation remains difficult for slang, idioms, and culturally dependent expressions.
- BLEU and chrF++ do not fully capture conversational naturalness or adequacy.
- The current training configuration uses one epoch and a small effective dataset in the recorded run.
- The official IN22-Conv evaluation is included as an optional extension in the notebook but was not part of the reported full-validation result.

---

## 16. Future Improvements

- Train on a larger cleaned conversational corpus.
- Evaluate on the official **IN22-Conv** benchmark.
- Compare additional LoRA ranks and target modules.
- Tune learning rate, number of epochs, and effective batch size.
- Add semantic evaluation alongside BLEU/chrF++.
- Build a dedicated conversational test suite for slang, idioms, contractions, and code-switching.
- Compare LoRA against full fine-tuning or other PEFT methods.
- Analyze performance by sentence length and conversational phenomenon.

---

## 17. Conclusion

This project demonstrates a complete domain-adaptation workflow for English–Hindi neural machine translation:

**conversational data → cleaning/filtering → IndicTrans2 preprocessing → baseline evaluation → LoRA fine-tuning → independent checkpoint evaluation → quantitative and qualitative analysis**

On the full 1,000-sentence held-out validation set, LoRA improved BLEU from **13.30 to 16.61** and chrF++ from **33.49 to 36.97**.

The project also highlights an important practical lesson: careful target-side preprocessing and independent evaluation are as important as the fine-tuning configuration itself.
