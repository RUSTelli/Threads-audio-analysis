from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa, os, csv
import torch

def audio_to_text(file_path):

    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="italian", task="transcribe")

    input_audio, sampling_rate = librosa.load(file_path, sr=16000)

    # Feature extraction
    input_features = processor(input_audio, sampling_rate=16000, return_tensors="pt").input_features

    # Token ID generation
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Output the text of audio
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)

def audio_folder_to_csv_text(folder_path, output_csv_path):

    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="italian", task="transcribe")

     # Open CSV on write mode
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Transcription'])
        
        # Scans files in the folder
        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)
            
            # Read audio
            input_audio, sampling_rate = librosa.load(file_path, sr=16000)
            
            # Feature extraction
            input_features = processor(input_audio, sampling_rate=16000, return_tensors="pt").input_features
            
            # Token ID generation
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            
            # Encoding token id to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # Write text in csv file
            writer.writerow([filename, transcription])

    # Output csv 
    return writer


    





