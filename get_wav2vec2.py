from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer
import torch


model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name_or_path, device)

processor = AutoProcessor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).to(device)