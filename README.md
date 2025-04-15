# Keyword Spotting in Persian Speech using wav2vec2

## 📌 Overview

This project focuses on implementing a **Keyword Spotting (KWS)** system for **Persian (Farsi)** conversational speech using a fine-tuned version of **wav2vec2-xlsr-large**. The system is capable of:

- Automatically transcribing spoken Persian
- Detecting the presence of predefined keywords in real-time audio
- Working effectively in noisy, real-world conditions such as call center conversations

The wav2vec2 model is trained using a dataset of Persian audio recordings, with optional support for n-gram language modeling to improve transcription quality.

---

## 🧠 What is Keyword Spotting?

**Keyword Spotting** refers to identifying specific words or phrases within continuous speech. It’s used in many applications like:

- Voice assistants (e.g., “Hey Siri” or “OK Google”)
- Customer service automation
- Surveillance and compliance monitoring

This project implements keyword spotting through ASR-based transcription followed by a search over transcribed text.

---

## 🗂️ Project Structure

The repository includes the following key files:

- `FINAL_TEST.py`: Main script for generating keyword spotting predictions on test audio files.
- `get_wav2vec2.py`: Functions to load and configure the Wav2Vec2 model.
- `load_model.py`: Loads the fine-tuned model for inference.
- `MCI_wav2vec2_train_V2.ipynb`: Notebook used for training and experimenting with the model.
- `preprocess.py`: Preprocessing routines for preparing audio input.
- `read_data.py`: Functions to handle dataset loading.
- `test.py`: Script to evaluate model performance.
- `train.py`: Script for training the Wav2Vec2 model on custom data.

---

## 🛠️ Features

- 🔊 **Speech Recognition** using wav2vec2-xlsr-large
- 🔍 **Keyword Detection** from ASR transcripts
- 📉 **Language Model Decoding** (KenLM for beam search decoding)
- 📊 **Evaluation Tools** for WER, precision, recall, and F1-score
- 🧪 Easily extensible for other Persian speech tasks

---

The trained ASR model is available on Hugging Face:  
👉 [aliyzd95/wav2vec2-large-xlsr-persian-KWS](https://huggingface.co/aliyzd95/wav2vec2-large-xlsr-persian-KWS)


## 📈 Evaluation Metrics

- **ASR Accuracy:** Measured using **Word Error Rate (WER)** via `jiwer`
- **Keyword Spotting Metrics:**
  - **Precision** = TP / (TP + FP)
  - **Recall** = TP / (TP + FN)
  - **F1-score** = Harmonic mean of precision and recall

---

## 🧪 Example Output


The script reads all `.wav` files from the input directory and generates a JSON file in the output directory with the format:

```json
{
  "files": [
    {
      "file-path": "/app/input/00711.wav",
      "outputs": {
        "occurence_vector": [0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "occurence_details": [
          {
            "word_name": "هدیه",
            "occurences": [
              {
                "start_time": "00:00:02:1234",
                "end_time": "00:00:03:1534",
                "confidence": 0.42
              },
              {
                "start_time": "00:00:05:5214",
                "end_time": "00:00:06:5512",
                "confidence": 0.96
              }
            ]
          }
        ]
      }
    }
  ]
}
```

---

## 🔒 Disclaimer

This project is intended for research and educational use only. The dataset used for fine-tuning is private and cannot be redistributed.

---

## 👤 Author

Developed by a graduate researcher in the field of **Artificial Intelligence and Speech Processing**.

---
