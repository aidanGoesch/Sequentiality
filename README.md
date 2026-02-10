# Sequentiality

This repository contains code for Sequentiality, a measure of contextual information gain

## Getting Started

To install the required dependencies run:

```bash
pip install -r requirements.txt
```

*IMPORTANT*
Make sure to create a `.env` and to add a valid HuggingFace API key. `src/keys.py` loads the API key from `.env`. Below is a sample `.env` to fill in:

```
HUGGING_FACE_API_KEY="your api key here"
```