import os
import sys
from pydub import AudioSegment
import whisperx
import tempfile

def srt_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def read_segments(path):
    segments = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) != 3: continue
            start, end, speaker = parts
            segments.append({
                'start': float(start),
                'end': float(end),
                'speaker': speaker.strip()
            })
    return segments

def main():
    audio_path = 'inputaudio.wav'
    diar_path = 'diarization_segments.txt'
    srt_path = 'inputvideo.srt'
    segments = read_segments(diar_path)
    audio = AudioSegment.from_wav(audio_path)
    model = whisperx.load_model('large-v2', device='cpu', compute_type='int8')
    with open(srt_path, 'w') as srt:
        for idx, seg in enumerate(segments, 1):
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            duration = (end_ms - start_ms) / 1000.0
            if duration < 0.5:
                print(f"Skipping segment {idx}: too short ({duration:.2f}s)")
                continue
            seg_audio = audio[start_ms:end_ms]
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio:
                seg_audio.export(temp_audio.name, format='wav')
                file_size = os.path.getsize(temp_audio.name)
                print(f"Segment {idx}: {seg['speaker']} {duration:.2f}s, file size {file_size} bytes")
                result = model.transcribe(temp_audio.name)
                print(f"WhisperX result for segment {idx}: {result}")
                if 'segments' in result and result['segments']:
                    text = ' '.join([seg['text'].strip() for seg in result['segments'] if seg['text'].strip()])
                else:
                    text = ''
                if not text:
                    print(f"Warning: Empty transcription for segment {idx}")
            srt.write(f"{idx}\n")
            srt.write(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n")
            srt.write(f"{seg['speaker']}: {text}\n\n")
    print(f"SRT file generated: {srt_path}")

if __name__ == '__main__':
    main()
