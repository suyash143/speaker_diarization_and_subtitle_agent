# Speaker Diarization and Quality Analysis System

## Overview
This system provides a complete pipeline for speaker diarization, subtitle generation, and quality analysis of audio/video content. It combines state-of-the-art ML models with comprehensive quality checks to deliver accurate speaker-labeled transcriptions.

## Features
- **Speaker Diarization**: Identifies and segments different speakers in audio
- **Voice Activity Detection**: Enhanced detection of speaker transitions
- **Automatic Transcription**: Generates transcriptions using WhisperX
- **Quality Analysis**: Comprehensive evaluation of diarization quality
- **Multi-format Support**: Handles both audio (.wav) and video (.mp4) inputs
- **SRT Generation**: Creates speaker-labeled subtitle files

## Prerequisites
- Python 3
- FFmpeg installed for video processing
- CUDA-compatible GPU
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/suyash143/speaker_diarization_and_subtitle_agent
cd speaker_diarization_and_subtitle_agent
touch .env
#add this in .env HUGGINGFACE_TOKEN=your_token_here

# ```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up Hugging Face authentication:
- Add your Hugging Face token in .env: `HUGGINGFACE_TOKEN=your_token_here`


## Usage

### Quick Start
Process an audio or video file with a single command:
```bash
./run.sh input_file.mp4  
# or
./run.sh input_file.wav  
```

### Individual Components

1. Speaker Diarization:
```bash
python diarize.py inputaudio.wav
```

2. Generate SRT:
```bash
python generate_srt.py input_file diarization_segments.txt
```

3. Quality Analysis:
```bash
python quality_check_agent.py
```

## Output Files
- **diarization_segments.txt**: Raw diarization output with timestamps
- **[Source filename].srt**: Generated subtitles with speaker labels
- **quality_check_results.json**: Detailed quality analysis report

## Quality Metrics
The system evaluates diarization quality across multiple dimensions:
- Speaker turn accuracy
- Label consistency
- Segmentation quality
- Audio clarity
- Speaker confusion detection

## System Components

### 1. Diarizer (`diarize.py`)
- Uses pyannote.audio for initial diarization
- Implements voice activity detection
- Includes segment refinement based on acoustic changes
- Handles variable-length segment optimization

### 2. SRT Generator (`generate_srt.py`)
- Combines diarization data with WhisperX transcription
- Generates formatted SRT files
- Includes speaker labels in subtitles

### 3. Quality Checker (`quality_check_agent.py`)
- Performs quality analysis
- Provides detailed feedback
- Generates confidence scores for each segment

## Configuration

### Audio Processing
- Sample Rate: 16kHz
- Format: WAV (mono)
- Supported Video Input: MP4

### Diarization Parameters
- Minimum segment duration: 1.0s
- Maximum segment duration: 6.0s
- Voice activity detection thresholds are adjustable

## Troubleshooting

### Error Messages
- Check the Hugging Face token if authentication fails
- Verify FFmpeg installation for video processing
- Ensure proper file permissions for output files

## Performance Optimization
- Use GPU acceleration when available
- Adjust batch **sizes** based on available memory

