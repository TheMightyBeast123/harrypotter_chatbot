# 🧙‍♂️ Harry Potter Chatbot with GPT2

A domain-specific chatbot fine-tuned on the **Harry Potter book series**, powered by a custom GPT2-style model using Hugging Face Transformers. Train, fine-tune, the project has been tested on google collab 

---

## 🧰 Features

- Fine-tunes a small GPT2 model on Harry Potter books or movie scripts
- Chat interactively via command line
- Cleaned and tokenized custom dataset
- Training and inference in **Google Colab** with GPU

---

## 🗂️ Directory Structure
harry_potter_chatbot/
├── data/ # Raw + tokenized dataset
├── preprocessing/ # Dataset preparation
├── model/ # Config + training scripts
├── chatbot/ # CLI chatbot
├── app/ # Optional web app
├── utils/ # Helpers
├── README.md
└── requirements.txt
## 🧹 Requirements

```bash
pip install torch transformers flask faiss-cpu
