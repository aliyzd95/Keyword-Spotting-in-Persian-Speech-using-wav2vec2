import os
from transformers import Wav2Vec2CTCTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor

save_dir = "ackerman/wav2vec2-large-xlsr-persian-MCI"
model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"

last_checkpoint = None
if os.path.exists(save_dir):
    last_checkpoint = get_last_checkpoint(save_dir)

print(last_checkpoint if last_checkpoint else str(None))

if not os.path.exists(save_dir) and not model_name_or_path:
    print("Load from scratch")
    tokenizer = Wav2Vec2CTCTokenizer(
        "./fa.vocab.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False
    )
else:
    print(f"Load from {model_name_or_path}")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name_or_path)

text = "از مهمونداری کنار بکشم"
print(" ".join(tokenizer.tokenize(text)))
print(tokenizer.decode(tokenizer.encode(text)))

if not os.path.exists(save_dir) and not model_name_or_path:
    print("Load from scratch")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
else:
    print(f"Load from {model_name_or_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)

if not os.path.exists(save_dir) and not model_name_or_path:
    print("Load from scratch")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
else:
    print(f"Load from {model_name_or_path}")
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)

if len(processor.tokenizer.get_vocab()) == len(processor.tokenizer):
    print(len(processor.tokenizer))

if not os.path.exists(save_dir):
    print("Saving ...")
    processor.save_pretrained(save_dir)
    print("Saved!")