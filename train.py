import os
import pickle
import random

import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_metric
from transformers import Wav2Vec2Processor, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


processor = Wav2Vec2Processor.from_pretrained("ackerman/wav2vec2-large-xlsr-persian-MCI")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    if isinstance(label_str, list):
        if isinstance(pred_str, list) and len(pred_str) == len(label_str):
            for index in random.sample(range(len(label_str)), 3):
                print(f'reference: "{label_str[index]}"')
                print(f'predicted: "{pred_str[index]}"')

        else:
            for index in random.sample(range(len(label_str)), 3):
                print(f'reference: "{label_str[index]}"')
                print(f'predicted: "{pred_str}"')

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Wav2Vec2ForCTC

last_checkpoint = None
if os.path.exists("ackerman/wav2vec2-large-xlsr-persian-MCI"):
    last_checkpoint = get_last_checkpoint("ackerman/wav2vec2-large-xlsr-persian-MCI")

print(last_checkpoint if last_checkpoint else str(None))
print(processor.tokenizer.get_vocab())
model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
model = Wav2Vec2ForCTC.from_pretrained(
    # "facebook/wav2vec2-large-xlsr-53" if not last_checkpoint else last_checkpoint,
    model_name_or_path if not last_checkpoint else last_checkpoint,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    bos_token_id=processor.tokenizer.bos_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer.get_vocab())
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="ackerman/wav2vec2-large-xlsr-persian-MCI",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=0.5,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=3e-5,
    warmup_steps=500,
    save_total_limit=2,
)

with open('ackerman/common_voice_train.pickle', 'rb') as tr:
    _common_voice_train = pickle.load(tr)

with open('ackerman/common_voice_test.pickle', 'rb') as te:
    _common_voice_test = pickle.load(te)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=_common_voice_train,
    eval_dataset=_common_voice_test,
    tokenizer=processor.feature_extractor,
)

print(f"last_checkpoint: {last_checkpoint}")
if last_checkpoint:
    print(f"last_checkpoint: {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    train_result = trainer.train()

metrics = train_result.metrics
max_train_samples = len(_common_voice_train)
metrics["train_samples"] = min(max_train_samples, len(_common_voice_train))

trainer.save_model()

trainer.log_metrics("train", metrics=metrics)
trainer.save_metrics("train", metrics=metrics)
trainer.save_state()



results = {}
metrics = trainer.evaluate()
max_val_samples = len(_common_voice_test)
metrics["eval_samples"] = min(max_val_samples, len(_common_voice_test))

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
