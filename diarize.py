print("Starting diarization script...")
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import whisperx
import numpy as np
import librosa

AUDIO_FILE = "inputaudio.wav"

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

try:
    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    if pipeline is None:
        raise RuntimeError("Pipeline could not be loaded. Check your Hugging Face token and permissions. Make sure you have accepted user conditions for all required models at https://hf.co/pyannote/speaker-diarization and https://hf.co/pyannote/segmentation.")
    print("Pipeline loaded. Running diarization...")
    diarization = pipeline(AUDIO_FILE)
    segments = list(diarization.itertracks(yield_label=True))

    print(f"Initial diarization found {len(segments)} segments")
    for turn, _, speaker in segments[:5]:
        print(f"  {turn.start:.2f}-{turn.end:.2f}: {speaker} ({turn.end-turn.start:.1f}s)")
    if len(segments) > 5:
        print("  ...")

    def detect_voice_activity_changes(audio_file, hop_length=512, frame_length=2048):
        print("Analyzing audio for voice activity changes...")
        y, sr = librosa.load(audio_file, sr=16000)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3, hop_length=hop_length)
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
        change_points = []
        rms_diff = np.abs(np.diff(rms))
        energy_threshold = np.percentile(rms_diff, 85)
        spec_diff = np.abs(np.diff(spec_cent))
        spec_threshold = np.percentile(spec_diff, 85)
        mfcc_changes = np.sum([np.abs(np.diff(mfcc)) for mfcc in mfccs], axis=0)
        mfcc_threshold = np.percentile(mfcc_changes, 85)
        for i in range(1, len(times)-1):
            indicators = 0
            if i < len(rms_diff) and rms_diff[i] > energy_threshold:
                indicators += 1
            if i < len(spec_diff) and spec_diff[i] > spec_threshold:
                indicators += 1
            if i < len(mfcc_changes) and mfcc_changes[i] > mfcc_threshold:
                indicators += 1
            if indicators >= 2:
                change_points.append(times[i])
        return change_points

    def refine_diarization_segments(original_segments, change_points, min_segment=1.0, max_segment=6.0):
        print(f"Refining segments using {len(change_points)} detected change points...")
        refined_segments = []
        for start_time, end_time, speaker in original_segments:
            duration = end_time - start_time
            if min_segment <= duration <= max_segment:
                refined_segments.append((start_time, end_time, speaker))
                continue
            if duration > max_segment:
                segment_changes = [cp for cp in change_points if start_time < cp < end_time]
                if not segment_changes:
                    num_chunks = int(duration / max_segment) + 1
                    chunk_duration = duration / num_chunks
                    for i in range(num_chunks):
                        chunk_start = start_time + i * chunk_duration
                        chunk_end = min(end_time, chunk_start + chunk_duration)
                        if chunk_end - chunk_start >= min_segment:
                            refined_segments.append((chunk_start, chunk_end, speaker))
                else:
                    current_start = start_time
                    for change_point in segment_changes:
                        if change_point - current_start >= min_segment:
                            refined_segments.append((current_start, change_point, speaker))
                            current_start = change_point
                    if end_time - current_start >= min_segment:
                        refined_segments.append((current_start, end_time, speaker))
            elif duration < min_segment:
                refined_segments.append((start_time, end_time, speaker))
        return refined_segments

    change_points = detect_voice_activity_changes(AUDIO_FILE)
    print(f"Detected {len(change_points)} potential speaker change points")

    simple_segments = [(turn.start, turn.end, speaker) for turn, _, speaker in segments]

    refined_segments = refine_diarization_segments(simple_segments, change_points)

    print(f"\nRefined diarization ({len(refined_segments)} segments):")
    for i, (start, end, speaker) in enumerate(refined_segments[:5]):
        duration = end - start
        print(f"{start:.2f} - {end:.2f}: {speaker} ({duration:.1f}s)")
        if duration > 8.0:
            print(f"  ⚠️  Long segment detected - may contain multiple speakers")

    with open("diarization_segments.txt", "w") as f:
        for start, end, speaker in refined_segments:
            f.write(f"{start:.2f},{end:.2f},{speaker}\n")
    with open("detected_changes.txt", "w") as f:
        for cp in change_points:
            f.write(f"{cp:.2f}\n")
    print(f"\nResults saved to diarization_segments.txt")
    print(f"Change points saved to detected_changes.txt")

    durations = [end - start for start, end, _ in refined_segments]
    print(f"\nSegment duration statistics:")
    print(f"  Average: {np.mean(durations):.1f}s")
    print(f"  Median: {np.median(durations):.1f}s")
    print(f"  Min: {min(durations):.1f}s")
    print(f"  Max: {max(durations):.1f}s")
    print(f"  Segments > 6s: {sum(1 for d in durations if d > 6)}")
except Exception as e:
    print(f"Error during diarization: {e}")
