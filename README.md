# Hate Speech Detection in Audio Content

## Authors
[Domenico Anzalone](https://github.com/DomenicoAnzalone), [Francesco Rastelli](https://github.com/RUSTelli)

## Description
This project focuses on developing a methodology for detecting hate speech in audio content on social media using machine learning approaches. We leverage language-specific models based on DistilBERT and integrate automatic audio transcription with OpenAI’s Whisper to create an efficient pipeline for hate speech detection and classification.

## Characteristics
- **Audio Transcription**: Utilizes OpenAI's Whisper for accurate transcription of audio content.
- **Sentiment Classification**: Employs fine-tuned DistilBERT models to classify sentiments in the transcribed text.
- **Hate-Type Classification**: Uses hierarchical classification strategies to identify specific types of hate speech.
- **Language-Specific Models**: Benchmarks models in five languages (Italian, English, French, Spanish, German) with high accuracy ranging from 90% to 95%.

## Prerequisites
- Python 3.11
- Other dependencies as listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/your_username/hate-speech-detection.git

# Navigate to the project directory
cd hate-speech-detection

# Install the dependencies
pip install -r requirements.txt
```

## License
This project is distributed under the [The Unlicense](LICENSE).
<p align='left'> 
    <img width="100" src="https://github.com/DomenicoAnzalone/FakeReviewDetection/assets/81223389/cfc25399-b043-4a7f-8029-79fc1cad2e45">
</p>
