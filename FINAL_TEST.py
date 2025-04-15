from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, Wav2Vec2ProcessorWithLM, \
    Wav2Vec2CTCTokenizer, pipeline, AutomaticSpeechRecognitionPipeline
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from datasets import load_metric, Dataset
import os
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Union
from datetime import datetime, timedelta

from transformers.pipelines.automatic_speech_recognition import _find_longest_common_sequence


class KWS_pipeline(AutomaticSpeechRecognitionPipeline):
    def postprocess(
            self, model_outputs, decoder_kwargs: Optional[Dict] = None, return_timestamps=None, return_language=None
    ):
        # Optional return types
        optional = {}

        if return_timestamps and self.type == "seq2seq":
            raise ValueError("We cannot return_timestamps yet on non-ctc models apart from Whisper !")
        if return_timestamps == "char" and self.type == "ctc_with_lm":
            raise ValueError("CTC with LM cannot return `char` timestamps, only `words`")
        if return_timestamps in {"char", "words"} and self.type == "seq2seq_whisper":
            raise ValueError("Whisper cannot return `char` nor `words` timestamps, use `True` instead.")

        if return_language is not None and self.type != "seq2seq_whisper":
            raise ValueError("Only whisper can return language for now.")

        final_items = []
        key = "logits" if self.type == "ctc_with_lm" else "tokens"
        stride = None
        for outputs in model_outputs:
            items = outputs[key].numpy()
            stride = outputs.get("stride", None)
            if stride is not None and self.type in {"ctc", "ctc_with_lm"}:
                total_n, left, right = stride
                # Total_n might be < logits.shape[1]
                # because of padding, that's why
                # we need to reconstruct this information
                # This won't work with left padding (which doesn't exist right now)
                right_n = total_n - right
                items = items[:, left:right_n]
            final_items.append(items)

        if stride and self.type == "seq2seq":
            items = _find_longest_common_sequence(final_items, self.tokenizer)
        elif self.type == "seq2seq_whisper":
            time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
            # Send the chunking back to seconds, it's easier to handle in whisper
            sampling_rate = self.feature_extractor.sampling_rate
            for output in model_outputs:
                if "stride" in output:
                    chunk_len, stride_left, stride_right = output["stride"]
                    # Go back in seconds
                    chunk_len /= sampling_rate
                    stride_left /= sampling_rate
                    stride_right /= sampling_rate
                    output["stride"] = chunk_len, stride_left, stride_right

            text, optional = self.tokenizer._decode_asr(
                model_outputs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                time_precision=time_precision,
            )
        else:
            items = np.concatenate(final_items, axis=1)
            items = items.squeeze(0)

        if self.type == "ctc_with_lm":
            if decoder_kwargs is None:
                decoder_kwargs = {}
            beams = self.decoder.decode_beams(items, **decoder_kwargs)
            text = beams[0][0]
            if return_timestamps:
                # Simply cast from pyctcdecode format to wav2vec2 format to leverage
                # pre-existing code later
                chunk_offset = beams[0][2]
                offsets = []
                for word, (start_offset, end_offset) in chunk_offset:
                    offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
        elif self.type != "seq2seq_whisper":
            skip_special_tokens = self.type != "ctc"
            text = self.tokenizer.decode(items, skip_special_tokens=skip_special_tokens)
            if return_timestamps:
                offsets = self.tokenizer.decode(
                    items, skip_special_tokens=skip_special_tokens, output_char_offsets=True
                )["char_offsets"]
                if return_timestamps == "word":
                    offsets = self.tokenizer._get_word_offsets(offsets, self.tokenizer.replace_word_delimiter_char)

        if return_timestamps and self.type not in {"seq2seq", "seq2seq_whisper"}:
            chunks = []
            for item in offsets:
                start = item["start_offset"] * self.model.config.inputs_to_logits_ratio
                start /= self.feature_extractor.sampling_rate

                stop = item["end_offset"] * self.model.config.inputs_to_logits_ratio
                stop /= self.feature_extractor.sampling_rate

                chunks.append({"text": item[return_timestamps], "timestamp": (start, stop)})
            optional["chunks"] = chunks

        extra = defaultdict(list)
        for output in model_outputs:
            output.pop("tokens", None)
            # output.pop("logits", None)
            output.pop("is_last", None)
            output.pop("stride", None)
            for k, v in output.items():
                extra[k].append(v)
        return {"text": text, **optional, **extra}


print('step 1 of 5: loading model!')

model_name_or_path = "ackerman/wav2vec2-large-xlsr-persian-KWS"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name_or_path, device)

processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).to(device)

print('step 1 of 5: done!')

keywords = ["بسته", "پیامک", "اینترنت", "خط", "هدیه", "سیمکارت", "قبض", "آنتن", "شارژ", "مکالمه", "ستاره", "مربع",
            "همراه من", "همراه اول", "گوشی", "مسدود", "تماس", "فعال", "غیرفعال"]

hotwords = ["بسته", "پیامک", "اینترنت", "خط", "هدیه", "سیمکارت", "قبض", "آنتن", "شارژ", "مکالمه", "ستاره", "مربع",
            "همراه من", "همراه اول", "همراه", "اول", "من", "گوشی", "مسدود", "تماس", "فعال", "غیر", "غیرفعال"]


def read_data(dataset_path='dataset'):
    files = []
    for file in os.listdir(dataset_path):
        file_path = f'{dataset_path}/{file}'
        files.append(file_path)
    df = pd.DataFrame(files, columns=['path'])
    return df


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(y=np.asarray(speech_array),
                                    orig_sr=sampling_rate,
                                    target_sr=processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def confidence_score(word_dict, index, logits):
    pred_ids = torch.argmax(logits, dim=-1)
    scores = torch.nn.functional.log_softmax(logits, dim=-1)
    pred_scores = scores.gather(1, pred_ids.unsqueeze(-1))[:, :, 0]
    probs = pred_scores[index, word_dict["start_offset"]: word_dict["end_offset"]]
    probs = torch.exp(probs)
    return round(torch.sum(probs).item() / (len(probs)), 4)


def predict(batch):
    features = processor(
        batch["speech"],
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    pipe = KWS_pipeline(model=model,
                        feature_extractor=processor.feature_extractor,
                        tokenizer=processor.tokenizer,
                        decoder=processor.decoder,
                        framework='pt',
                        device=0)

    output = pipe(batch["path"],
                  return_timestamps='word',
                  chunk_length_s=10,
                  stride_length_s=(0, 0)
                  )

    word_offsets = []

    for i, l in enumerate(output['logits']):
        logits = output['logits'][i]
        outputs = processor.batch_decode(logits.cpu().detach().numpy(), beam_width=400, output_word_offsets=True,
                                         hotwords=hotwords, hotword_weight=1.0
                                         )
        time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
        for d in outputs.word_offsets[0]:
            word_offsets.append(
                {
                    "word": d["word"],
                    "start_time": d["start_offset"] * time_offset + (i * 10),
                    "end_time": d["end_offset"] * time_offset + (i * 10),
                    "confidence": confidence_score(d, 0, logits)
                }
            )

    batch['word_offsets'] = word_offsets

    return batch


def format_time(sec):
    time = str(timedelta(seconds=sec))
    try:
        time = time.split('.')
        td = f'0{time[0]}:{time[1][:4]}'
    except:
        td = f'0{time[0]}:0000'
    return td


def generate_result(batch):
    word_offsets = batch['word_offsets']

    keywords_list = []
    for i in range(len(word_offsets)):
        word = word_offsets[i]['word']
        if word in keywords:
            word_dict = dict()
            word_dict['word_name'] = word
            word_dict['start_time'] = format_time(word_offsets[i]['start_time'])
            word_dict['end_time'] = format_time(word_offsets[i]['end_time'])
            word_dict['confidence'] = word_offsets[i]['confidence']
            keywords_list.append(word_dict)
        elif word == 'همراه' and 'من' in word_offsets[i + 1]['word']:
            word_dict = dict()
            word_dict['word_name'] = 'همراه من'
            word_dict['start_time'] = format_time(word_offsets[i]['start_time'])
            word_dict['end_time'] = format_time(word_offsets[i + 1]['end_time'])
            word_dict['confidence'] = word_offsets[i]['confidence']
            keywords_list.append(word_dict)
        elif word == 'همراه' and 'اول' in word_offsets[i + 1]['word']:
            word_dict = dict()
            word_dict['word_name'] = 'همراه اول'
            word_dict['start_time'] = format_time(word_offsets[i]['start_time'])
            word_dict['end_time'] = format_time(word_offsets[i + 1]['end_time'])
            word_dict['confidence'] = word_offsets[i]['confidence']
            keywords_list.append(word_dict)
        elif word == 'غیر' and 'فعال' in word_offsets[i + 1]['word']:
            word_dict = dict()
            word_dict['word_name'] = 'غیرفعال'
            word_dict['start_time'] = format_time(word_offsets[i]['start_time'])
            word_dict['end_time'] = format_time(word_offsets[i + 1]['end_time'])
            word_dict['confidence'] = word_offsets[i]['confidence']
            keywords_list.append(word_dict)

    occurence_details = []
    for kw in keywords_list:
        for oc in occurence_details:
            if kw['word_name'] == oc["word_name"]:
                oc['occurences'].append(
                    {'start_time': kw['start_time'], 'end_time': kw['end_time'], 'confidence': kw['confidence']}
                )
                break
        else:
            kw_dict = dict()
            kw_dict['word_name'] = kw['word_name']
            kw_dict['occurences'] = [
                {'start_time': kw['start_time'], 'end_time': kw['end_time'], 'confidence': kw['confidence']}
            ]
            occurence_details.append(kw_dict)

    occurence_vector = [0] * len(keywords)
    for i, kw in enumerate(keywords):
        for oc in occurence_details:
            if kw == oc['word_name']:
                occurence_vector[i] = len(oc['occurences'])

    outputs = {'occurence_vector': occurence_vector, 'occurence_details': occurence_details}

    batch['outputs'] = outputs

    return batch


def all_files_result(result_json):
    all_files = []
    for data in result_json:
        file_dict = dict()
        file_dict['file_path'] = data['path']
        file_dict['outputs'] = data['outputs']
        all_files.append(file_dict)

    return all_files


def generate_json(all_files, output_path):
    output_ackerman = dict()
    output_ackerman['files'] = all_files
    with open(f'{output_path}/output_ackerman.json', 'w') as fp:
        json.dump(output_ackerman, fp, indent=2)


print('step 2 of 5: reading dataset')

dataset_path = '/content/dataset'
df = read_data(dataset_path=dataset_path)
dataset = Dataset.from_pandas(pd.DataFrame(data=df))

print('step 2 of 5: done!')
print('step 3 of 5: data preprocessing...')

dataset = dataset.map(speech_file_to_array_fn)

print('step 3 of 5: done!')
print('step 4 of 5: making predictions... be patient! :)')

result = dataset.map(predict)
result = result.remove_columns("speech")

print('step 4 of 5: done!')
print('step 4 of 5: generating result...')

result_json = result.map(generate_result, batched=False)
all_files = all_files_result(result_json)

print('step 4 of 5: done!')
print('step 5 of 5: creating json output...')

output_path = '/content'
generate_json(all_files, output_path)

print('step 5 of 5: done!')
print('*** generated result as json is now available at /output/output_ackerman.json ***')
