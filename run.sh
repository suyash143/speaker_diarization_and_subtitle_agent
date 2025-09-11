#!/bin/bash

# Usage: ./run.sh input_file
# Input can be either audio (wav) or video (mp4)

set -e

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    echo "Error: Please provide an input file (audio or video)"
    echo "Usage: ./run.sh input_file"
    exit 1
fi

# Get file extension
FILE_EXT="${INPUT_FILE##*.}"
FILENAME="${INPUT_FILE%.*}"

# Always use inputaudio.wav as our working audio file
AUDIO_FILE="inputaudio.wav"

# Convert input to audio if needed
if [[ "$FILE_EXT" == "mp4" ]]; then
    echo "Converting video to audio..."
    if ! ffmpeg -i "$INPUT_FILE" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_FILE" -y; then
        echo "Error: Failed to convert video to audio. Is ffmpeg installed?"
        exit 1
    fi
elif [[ "$FILE_EXT" == "wav" ]]; then
    echo "Copying audio file..."
    cp "$INPUT_FILE" "$AUDIO_FILE"
else
    echo "Error: Unsupported file format. Please provide .wav or .mp4 file"
    exit 1
fi

echo "Starting speaker diarization..."
if ! python3.10 diarize.py "$AUDIO_FILE"; then
    echo "Error: Diarization failed"
    exit 1
fi

echo "Generating SRT file..."
if ! python3.10 generate_srt.py "$INPUT_FILE" diarization_segments.txt; then
    echo "Error: SRT generation failed"
    exit 1
fi

echo "Running quality check..."
if ! python3.10 quality_check_agent.py; then
    echo "Error: Quality check failed"
    exit 1
fi

echo "All steps completed successfully!"
echo "Output files:"
echo "- inputaudio.wav: Extracted/converted audio"
echo "- diarization_segments.txt: Speaker diarization output"
echo "- ${FILENAME}.srt: Generated subtitles"
echo "- quality_check_results.json: Quality analysis results"
