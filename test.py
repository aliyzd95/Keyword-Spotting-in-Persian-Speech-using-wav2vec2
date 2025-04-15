from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer
import torch
import os
import json
import torchaudio
import librosa
import numpy as np
import pandas as pd
from datasets import load_metric, Dataset

model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name_or_path, device)

processor = AutoProcessor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).to(device)

vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())

vocab = []
for _, token in sort_vocab:
    vocab.append(token.lower())

vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    do_lower_case=False
)

from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=vocab,
    kenlm_model_path="/dataset/LM/w2c_cleaned.binary",
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=tokenizer,
    decoder=decoder
)

test_data = []
path = 'dataset/test'
for file in os.listdir(path):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        sample = {}
        sample['path'] = f'{path}/{filename}'
        sample['sentence'] = f'{path}/{filename[:-4]}.txt'
        test_data.append(sample)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(y=np.asarray(speech_array),
                                    orig_sr=sampling_rate,
                                    target_sr=processor_with_lm.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


dataset = Dataset.from_pandas(pd.DataFrame(data=test_data))
dataset = dataset.map(speech_file_to_array_fn)

import gc


def predict(batch):
    features = processor_with_lm(
        batch["speech"],
        sampling_rate=processor_with_lm.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    outputs = processor_with_lm.decode(logits, beam_width=400, output_word_offsets=True)

    time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
    word_offsets = [
        {
            "word": d["word"],
            "start_time": d["start_offset"] * time_offset,
            "end_time": d["end_offset"] * time_offset,
        }
        for d in outputs.word_offsets
    ]

    batch["word_offsets"] = word_offsets

    batch["predicted"] = outputs.text

    gc.collect()
    torch.cuda.empty_cache()

    return batch


result = dataset.map(predict, batched=False)

wer = load_metric("wer")
print(f"WER: {100 * wer.compute(predictions=result['predicted'], references=result['sentence'])}")
