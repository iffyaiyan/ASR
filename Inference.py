import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf

def transcribe_audio(audio_path, model_path):
    # Load fine-tuned model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Load audio file
    audio_input, sample_rate = sf.read(audio_path)
    
    # Process the audio
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the output
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

if __name__ == "__main__":
    audio_path = "path/to/your/audio/file.wav"   # give this path as per your system path, in fact path from the excel should also work
    model_path = "./wav2vec2-court-transcription-final"
    transcription = transcribe_audio(audio_path, model_path)
    print(f"Transcription: {transcription}")