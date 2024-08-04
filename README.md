# Court Transcription ASR Project

This project aims to create an Automatic Speech Recognition (ASR) system specifically tailored for court transcriptions. It includes data processing, dataset creation, model fine-tuning, and inference components.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Components](#components)
5. [Docker](#docker)
6. [Notes](#notes)

## Project Structure

```
court-transcription-asr/
│
├── data_processing.py
├── dataset_creation.py
├── model_finetuning.py
├── inference.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/iffyaiyan/ASR.githttps://github.com/iffyaiyan/ASR.git
   cd court-transcription-asr
   ```

2. Install Docker if you haven't already. Follow the official Docker documentation for your operating system.

3. (Optional) If you plan to use GPU acceleration, install NVIDIA Docker support.

4. Obtain a Google Service Account JSON key file for accessing Google Sheets API. Place this file in the project directory.

5. Update the `data_processing.py` file with the correct path to your service account JSON file.

## Usage

You can run each component of the project separately or use Docker to set up the entire environment.

### Running components individually:

1. Data Processing:
   ```
   python data_processing.py
   ```

2. Dataset Creation:
   ```
   python dataset_creation.py
   ```

3. Model Fine-tuning:
   ```
   python model_finetuning.py
   ```

4. Inference:
   ```
   python inference.py
   ```

### Using Docker:

1. Build the Docker image:
   ```
   docker build -t court-transcription-asr .
   ```

2. Run the container:
   ```
   docker run -it --gpus all -v /path/to/your/data:/app/data court-transcription-asr
   ```
   Replace `/path/to/your/data` with the actual path to your data directory on the host machine.

## Components

### 1. Data Processing (`data_processing.py`)

This script handles the initial data processing:
- Reads data from a Google Sheet
- Downloads audio files and transcripts
- Segments audio files into smaller chunks

### 2. Dataset Creation (`dataset_creation.py`)

This script creates a dataset suitable for ASR model training:
- Aligns audio segments with corresponding transcript portions
- Splits data into training and testing sets
- Saves the dataset in a format compatible with the Common Voice dataset

### 3. Model Fine-tuning (`model_finetuning.py`)

This script fine-tunes a Wav2Vec2 model on the created dataset:
- Loads and preprocesses the dataset
- Sets up the Wav2Vec2 model, tokenizer, and feature extractor
- Defines training arguments
- Trains the model
- Saves the fine-tuned model

### 4. Inference (`inference.py`)

This script uses the fine-tuned model to transcribe new audio files:
- Loads the fine-tuned model
- Processes input audio
- Generates and returns transcriptions

## Docker

The provided Dockerfile sets up an environment with all necessary dependencies. It uses Python 3.8 as the base image and installs required system and Python packages.

## Notes

- Ensure you have sufficient disk space for downloaded audio files and the processed dataset.
- Fine-tuning the model can be computationally intensive. A GPU is recommended for this step.
- The current setup uses the Wav2Vec2 model. You may experiment with other models like Whisper or Conformer by modifying the `model_finetuning.py` script.
- Always ensure you have the necessary rights and permissions to use the court transcription data.

## Future Improvements

- Implement more sophisticated audio-text alignment techniques
- Add data augmentation to increase dataset diversity
- Experiment with different ASR models and architectures
- Implement a web interface for easy transcription of new audio files
- Add support for batch processing of multiple audio files

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License

[MIT License](https://opensource.org/licenses/MIT)