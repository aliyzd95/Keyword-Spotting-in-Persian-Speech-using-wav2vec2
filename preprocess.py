import pickle
import random

import numpy as np
import torchaudio
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2Processor

target_sampling_rate = 16_000


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, target_sampling_rate)

    batch["speech"] = speech_array
    batch["sampling_rate"] = target_sampling_rate
    batch["duration_in_seconds"] = len(batch["speech"]) / target_sampling_rate
    batch["target_text"] = batch["cleaned_tweet"]
    return batch


common_voice_train = load_dataset("csv", data_files={"train": "dataset/csv/train.csv"}, delimiter=",")["train"]
common_voice_test = load_dataset("csv", data_files={"test": "dataset/csv/test.csv"}, delimiter=",")["test"]

common_voice_train = common_voice_train.map(speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)

print(common_voice_train[0]["sampling_rate"])
print(common_voice_test[0]["sampling_rate"])

sample = common_voice_train
rand_int = random.randint(0, len(sample))

print("Target text:", sample[rand_int]["target_text"])
print("Input array shape:", np.asarray(sample[rand_int]["speech"]).shape)
print("Sampling rate:", sample[rand_int]["sampling_rate"])

processor = Wav2Vec2Processor.from_pretrained("ackerman/wav2vec2-large-xlsr-persian-MCI")


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
            len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names,
                                             batch_size=4, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names,
                                           batch_size=4, batched=True)

with open('ackerman/common_voice_train.pickle', 'wb') as train:
    pickle.dump(common_voice_train, train, protocol=pickle.HIGHEST_PROTOCOL)

with open('ackerman/common_voice_test.pickle', 'wb') as test:
    pickle.dump(common_voice_test, test, protocol=pickle.HIGHEST_PROTOCOL)
