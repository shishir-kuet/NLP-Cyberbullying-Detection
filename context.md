# Project Context: Bangla Cyberbullying Detection and Social Engagement Prediction

## 1. PROJECT OVERVIEW

This is an NLP research/project developed using a Bangla social-media cyberbullying dataset.

Project title:

"Bangla Cyberbullying Detection and Social Engagement Prediction Using TF-IDF and BanglaBERT"

Alternative academic title:

"A Comparative Study of TF-IDF-Based and BanglaBERT Models for Bangla Cyberbullying Detection and Social Engagement Prediction"

The project focuses on two NLP tasks:

1. Multiclass Bangla cyberbullying classification
2. Social engagement/reaction-count regression

The primary model comparison is:

- TF-IDF + Logistic Regression
- BanglaBERT

LSTM is OPTIONAL and should only be added later if explicitly requested.

Do not unnecessarily expand the project with additional models such as XLM-R, mBERT, RoBERTa, etc. The current planned Transformer model is BanglaBERT only.

---

# 2. RESEARCH OBJECTIVES

The main objectives are:

### Objective 1 — Cyberbullying Classification

Given a Bangla social-media comment, classify it into one of five labels:

- Neutral
- Harassment
- Sexual Aggression
- Hate Speech
- Violent Extremism

This is a 5-class multiclass classification problem.

### Objective 2 — Social Engagement Prediction

Given the text of a Bangla social-media comment, predict its:

`Comment React Number`

This is a regression problem.

### Objective 3 — Model Comparison

Compare a traditional NLP pipeline:

TF-IDF + Logistic Regression

against a contextual Transformer-based pipeline:

BanglaBERT

The comparison should be systematic and use the same train/validation/test split and evaluation protocol.

### Objective 4 — Error Analysis

Analyze where and why the models make mistakes, especially for difficult Bangla social-media expressions.

### Objective 5 — Optional Future Extension

If time permits, add an LSTM/BiLSTM model as an additional deep-learning baseline.

Do NOT implement this in the initial phase.

---

# 3. DATASET

Dataset filename:

`Cyberbulling Bangla Dataset(1).csv`

The dataset contains approximately 44,001 Bangla social-media comments.

Columns:

1. `Text`
2. `Category`
3. `Gender`
4. `Comment React Number`
5. `Label`

### Column meanings

#### Text

The actual Bangla social-media comment.

This is the primary textual input.

#### Category

The category/context of the target/person associated with the comment.

Examples include categories such as:

- Actor
- Singer
- Politician
- Sports
- Social

This should NOT be used as an input feature in the primary text-only experiment.

It may be analyzed during EDA.

#### Gender

Gender-related metadata.

Do NOT use it as an input feature in the primary model.

It can be analyzed separately during EDA if useful.

#### Comment React Number

Numerical social engagement/reaction count.

This is the regression target.

#### Label

Cyberbullying classification target.

Five classes:

- Neutral
- Harassment
- Sexual Aggression
- Hate Speech
- Violent Extremism

---

# 4. IMPORTANT DATASET OBSERVATIONS

The dataset contains duplicate text values.

There are also cases where identical text can occur with different labels.

This is extremely important because careless random splitting can cause data leakage or artificially inflated evaluation results.

Before model training:

1. Inspect missing values.
2. Inspect duplicate rows.
3. Inspect duplicate `Text`.
4. Identify duplicate Text with the same label.
5. Identify duplicate Text with conflicting labels.
6. Inspect very short/empty comments.
7. Analyze reaction-count distribution.
8. Analyze class imbalance.
9. Analyze text length.

Do NOT silently remove conflicting-label duplicates.

The duplicate-handling strategy must be explicitly documented.

Do not fabricate statistics. Always calculate them directly from the CSV.

---

# 5. PRIMARY PROJECT DESIGN

The project should follow this architecture:

Raw Dataset
    ↓
Data Validation
    ↓
Duplicate / Conflict Analysis
    ↓
EDA
    ↓
Conservative Bangla Text Preprocessing
    ↓
Train / Validation / Test Split
    ↓
    ┌──────────────────────────┐
    │                          │
    ↓                          ↓
TF-IDF                    BanglaBERT
    ↓                          ↓
Logistic Regression       Fine-tuning
    ↓                          ↓
Classification            Classification
    │                          │
    └──────────────┬───────────┘
                   ↓
          Model Comparison
                   ↓
          Error Analysis


Regression branch:

Text
 ↓
TF-IDF
 ↓
Ridge Regression
 ↓
Reaction Count


Text
 ↓
BanglaBERT
 ↓
Regression Head
 ↓
Reaction Count