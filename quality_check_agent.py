import csv
import json
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class EnhancedQualityCheckAgent:
    def __init__(self, diarization_file: str, audio_file: str = None, transcript_file: str = None):
        self.diarization_file = diarization_file
        self.audio_file = audio_file
        self.transcript_file = transcript_file
        self.segments = []
        self.audio_features = None
        self.transcript_data = None

        self.load_segments()
        if self.audio_file:
            self.extract_audio_features()
        if self.transcript_file:
            self.load_transcript()

    def load_segments(self):
        try:
            with open(self.diarization_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    if ',' in line:
                        parts = line.split(',')
                    else:
                        parts = line.split()
                    if len(parts) >= 3:
                        try:
                            start = float(parts[0])
                            end = float(parts[1])
                            speaker = parts[2].strip()
                            text = parts[3].strip() if len(parts) > 3 else ""
                            self.segments.append({
                                'start': start,
                                'end': end,
                                'speaker': speaker,
                                'text': text,
                                'duration': end - start
                            })
                        except ValueError as e:
                            print(f"Warning: Could not parse line {line_num}: {line} - {e}")
                            continue
                    else:
                        print(f"Warning: Invalid format on line {line_num}: {line}")
            print(f"Loaded {len(self.segments)} segments from {self.diarization_file}")
            if self.segments:
                speakers = set(seg['speaker'] for seg in self.segments)
                print(f"Detected speakers: {', '.join(sorted(speakers))}")
        except Exception as e:
            print(f"Error loading segments: {e}")
            raise

    def extract_audio_features(self):
        try:
            print("Extracting audio features...")
            y, sr = librosa.load(self.audio_file, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr * 0.25))
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=int(sr * 0.25))
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=int(sr * 0.25))
            features = np.vstack([mfcc, spectral_centroids, spectral_rolloff])
            self.audio_features = {
                'features': features,
                'times': librosa.frames_to_time(np.arange(features.shape[1]), sr=sr, hop_length=int(sr * 0.25)),
                'sr': sr,
                'audio': y
            }
            print(f"Extracted audio features: {features.shape}")
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            self.audio_features = None

    def load_transcript(self):
        try:
            with open(self.transcript_file, 'r') as f:
                self.transcript_data = f.read()
        except Exception as e:
            print(f"Error loading transcript: {e}")
            self.transcript_data = None

    def analyze_speaker_turn_accuracy(self) -> Dict:
        """
        Analyze speaker turn accuracy using audio features
        """
        if not self.audio_features:
            return {'score': 0.5, 'feedback': 'Audio analysis unavailable', 'details': []}

        print("Analyzing speaker turn accuracy...")
        turn_scores = []
        details = []

        for i in range(len(self.segments) - 1):
            current_seg = self.segments[i]
            next_seg = self.segments[i + 1]

            if current_seg['speaker'] != next_seg['speaker']:
                transition_time = current_seg['end']
                window_size = 1.0
                start_idx = max(0, int((transition_time - window_size / 2) * 4))
                end_idx = min(len(self.audio_features['times']), int((transition_time + window_size / 2) * 4))

                if end_idx > start_idx + 2:
                    before_features = self.audio_features['features'][:, start_idx:int((transition_time) * 4)]
                    after_features = self.audio_features['features'][:, int((transition_time) * 4):end_idx]

                    if before_features.shape[1] > 0 and after_features.shape[1] > 0:
                        before_mean = np.mean(before_features, axis=1)
                        after_mean = np.mean(after_features, axis=1)
                        similarity = cosine_similarity([before_mean], [after_mean])[0][0]
                        turn_score = 1 - similarity

                        turn_scores.append(max(0, min(1, turn_score)))

                        details.append({
                            'transition_time': transition_time,
                            'from_speaker': current_seg['speaker'],
                            'to_speaker': next_seg['speaker'],
                            'turn_score': turn_score,
                            'confidence': 'High' if turn_score > 0.3 else 'Medium' if turn_score > 0.15 else 'Low'
                        })

        overall_score = np.mean(turn_scores) if turn_scores else 0.5

        feedback = f"Speaker turn accuracy: {overall_score:.2f}. "
        if overall_score > 0.25:
            feedback += "Good speaker boundary detection."
        elif overall_score > 0.15:
            feedback += "Moderate speaker boundary detection."
        else:
            feedback += "Weak speaker boundary detection - consider improving diarization."

        return {
            'score': overall_score,
            'feedback': feedback,
            'details': details,
            'num_transitions': len(turn_scores)
        }

    def analyze_speaker_consistency(self) -> Dict:
        """
        Analyze speaker consistency using voice embeddings and clustering
        """
        if not self.audio_features:
            return {'score': 0.5, 'feedback': 'Audio analysis unavailable', 'details': {}}

        print("Analyzing speaker consistency...")

        speaker_segments = {}
        for seg in self.segments:
            speaker = seg['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)

        consistency_scores = {}
        speaker_features = {}

        for speaker, segments in speaker_segments.items():
            if len(segments) < 2:
                consistency_scores[speaker] = 1.0
                continue

            segment_features = []
            for seg in segments:
                start_idx = int(seg['start'] * 4)
                end_idx = int(seg['end'] * 4)

                if end_idx > start_idx:
                    seg_features = self.audio_features['features'][:, start_idx:end_idx]
                    if seg_features.shape[1] > 0:
                        segment_features.append(np.mean(seg_features, axis=1))

            if len(segment_features) > 1:
                similarities = []
                for i in range(len(segment_features)):
                    for j in range(i + 1, len(segment_features)):
                        sim = cosine_similarity([segment_features[i]], [segment_features[j]])[0][0]
                        similarities.append(sim)

                consistency_score = np.mean(similarities)
                consistency_scores[speaker] = consistency_score
                speaker_features[speaker] = segment_features
            else:
                consistency_scores[speaker] = 1.0

        confusion_analysis = self._detect_speaker_confusion(speaker_features)

        overall_consistency = np.mean(list(consistency_scores.values()))

        feedback = f"Speaker consistency: {overall_consistency:.2f}. "
        if overall_consistency > 0.7:
            feedback += "Excellent speaker label consistency."
        elif overall_consistency > 0.5:
            feedback += "Good speaker label consistency."
        else:
            feedback += "Poor speaker consistency - possible label confusion."

        if confusion_analysis['potential_confusion']:
            feedback += f" Warning: Potential confusion between {confusion_analysis['confused_pairs']}."

        return {
            'score': overall_consistency,
            'feedback': feedback,
            'speaker_scores': consistency_scores,
            'confusion_analysis': confusion_analysis
        }

    def _detect_speaker_confusion(self, speaker_features: Dict) -> Dict:
        """Detect potential speaker confusion using clustering"""
        if len(speaker_features) < 2:
            return {'potential_confusion': False, 'confused_pairs': []}

        all_features = []
        speaker_labels = []

        for speaker, features in speaker_features.items():
            for feat in features:
                all_features.append(feat)
                speaker_labels.append(speaker)

        if len(all_features) < 4:
            return {'potential_confusion': False, 'confused_pairs': []}

        n_clusters = min(len(set(speaker_labels)), len(all_features) // 2)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(all_features)

            confused_pairs = []
            for cluster_id in range(n_clusters):
                cluster_speakers = [speaker_labels[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                unique_speakers = set(cluster_speakers)

                if len(unique_speakers) > 1:
                    confused_pairs.extend(list(unique_speakers))

            return {
                'potential_confusion': len(confused_pairs) > 0,
                'confused_pairs': confused_pairs
            }

        return {'potential_confusion': False, 'confused_pairs': []}

    def analyze_segmentation_quality(self) -> Dict:
        """
        Analyze over-segmentation and missed segments
        """
        print("Analyzing segmentation quality...")

        issues = []
        confidence = 1.0

        durations = [seg['duration'] for seg in self.segments]
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)

        very_short_segments = [seg for seg in self.segments if seg['duration'] < 0.5]
        if len(very_short_segments) > len(self.segments) * 0.2:
            issues.append(f"Possible over-segmentation: {len(very_short_segments)} very short segments")
            confidence -= 0.3

        very_long_segments = [seg for seg in self.segments if seg['duration'] > 15]
        if very_long_segments:
            issues.append(f"Possible missed transitions: {len(very_long_segments)} very long segments")
            confidence -= 0.2

        rapid_switches = 0
        for i in range(len(self.segments) - 1):
            if (self.segments[i]['speaker'] != self.segments[i + 1]['speaker'] and
                    self.segments[i]['duration'] < 1.0):
                rapid_switches += 1

        if rapid_switches > len(self.segments) * 0.15:
            issues.append(f"Many rapid speaker switches: {rapid_switches}")
            confidence -= 0.2

        gaps_overlaps = []
        for i in range(len(self.segments) - 1):
            gap = self.segments[i + 1]['start'] - self.segments[i]['end']
            if abs(gap) > 0.1:
                gaps_overlaps.append({
                    'type': 'gap' if gap > 0 else 'overlap',
                    'duration': abs(gap),
                    'position': self.segments[i]['end']
                })

        if len(gaps_overlaps) > len(self.segments) * 0.1:
            issues.append(f"Multiple gaps/overlaps detected: {len(gaps_overlaps)}")
            confidence -= 0.1

        feedback = "Good segmentation quality." if not issues else "; ".join(issues)

        return {
            'score': max(confidence, 0),
            'feedback': feedback,
            'statistics': {
                'avg_duration': avg_duration,
                'std_duration': std_duration,
                'total_segments': len(self.segments),
                'very_short_segments': len(very_short_segments),
                'very_long_segments': len(very_long_segments),
                'rapid_switches': rapid_switches,
                'gaps_overlaps': len(gaps_overlaps)
            },
            'issues': issues
        }

    def analyze_individual_segments(self) -> Dict:
        """
        Analyze each segment individually and provide confidence metrics and feedback
        """
        print("Analyzing individual segments...")

        segment_analyses = []

        for i, seg in enumerate(self.segments):
            analysis = {
                'segment_id': i + 1,
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['duration'],
                'speaker': seg['speaker'],
                'text': seg['text'],
                'confidence': 1.0,
                'issues': [],
                'feedback': ""
            }

            if seg['duration'] < 0.5:
                analysis['confidence'] -= 0.3
                analysis['issues'].append("Very short segment (possible over-segmentation)")
            elif seg['duration'] > 15:
                analysis['confidence'] -= 0.2
                analysis['issues'].append("Very long segment (possible missed speaker change)")

            if i > 0:
                prev_seg = self.segments[i - 1]
                gap = seg['start'] - prev_seg['end']

                if gap > 2.0:
                    analysis['confidence'] -= 0.1
                    analysis['issues'].append(f"Large gap ({gap:.1f}s) from previous segment")
                elif gap < -0.1:
                    analysis['confidence'] -= 0.2
                    analysis['issues'].append(f"Overlap ({abs(gap):.1f}s) with previous segment")

                if (seg['speaker'] != prev_seg['speaker'] and
                        prev_seg['duration'] < 1.0 and seg['duration'] < 1.0):
                    analysis['confidence'] -= 0.15
                    analysis['issues'].append("Rapid speaker switch (possible diarization error)")

            if i < len(self.segments) - 1:
                next_seg = self.segments[i + 1]
                gap = next_seg['start'] - seg['end']

                if gap > 2.0:
                    analysis['confidence'] -= 0.05
                    analysis['issues'].append(f"Large gap ({gap:.1f}s) to next segment")

            if self.audio_features:
                audio_confidence = self._analyze_segment_audio_quality(seg, i)
                analysis['confidence'] = (analysis['confidence'] + audio_confidence) / 2

                if audio_confidence < 0.3:
                    analysis['issues'].append("Low audio quality or unclear speech")
                elif audio_confidence < 0.6:
                    analysis['issues'].append("Moderate audio clarity")

            speaker_consistency = self._check_segment_speaker_consistency(seg, i)
            if speaker_consistency < 0.7:
                analysis['confidence'] -= 0.1
                analysis['issues'].append("Potential speaker inconsistency within segment")

            analysis['confidence'] = max(0.0, min(1.0, analysis['confidence']))

            if analysis['confidence'] > 0.8:
                quality = "High"
            elif analysis['confidence'] > 0.6:
                quality = "Good"
            elif analysis['confidence'] > 0.4:
                quality = "Fair"
            else:
                quality = "Poor"

            feedback_parts = [f"Quality: {quality}"]
            if analysis['issues']:
                feedback_parts.append("Issues: " + "; ".join(analysis['issues']))
            else:
                feedback_parts.append("No significant issues detected")

            analysis['feedback'] = ". ".join(feedback_parts)
            segment_analyses.append(analysis)

        return {
            'individual_segments': segment_analyses,
            'summary_stats': self._calculate_segment_summary_stats(segment_analyses)
        }

    def _analyze_segment_audio_quality(self, segment: Dict, segment_idx: int) -> float:
        """
        Analyze audio quality for a specific segment
        """
        try:
            start_frame = int(segment['start'] * 4)
            end_frame = int(segment['end'] * 4)

            if end_frame <= start_frame or start_frame >= self.audio_features['features'].shape[1]:
                return 0.5

            seg_features = self.audio_features['features'][:, start_frame:end_frame]

            if seg_features.shape[1] == 0:
                return 0.5

            feature_std = np.std(seg_features, axis=1)
            stability_score = 1.0 / (1.0 + np.mean(feature_std))

            energy = np.mean(seg_features[0, :])
            energy_score = min(1.0, max(0.0, (energy + 50) / 100))

            audio_confidence = (stability_score * 0.6 + energy_score * 0.4)

            return max(0.0, min(1.0, audio_confidence))

        except Exception:
            return 0.5

    def _check_segment_speaker_consistency(self, segment: Dict, segment_idx: int) -> float:
        """
        Check speaker consistency within a segment using audio features
        """
        if not self.audio_features or segment['duration'] < 1.0:
            return 1.0

        try:
            start_frame = int(segment['start'] * 4)
            end_frame = int(segment['end'] * 4)

            if end_frame <= start_frame:
                return 1.0

            seg_features = self.audio_features['features'][:, start_frame:end_frame]

            if seg_features.shape[1] < 8:
                return 1.0

            chunk_size = max(1, seg_features.shape[1] // 4)
            chunk_features = []

            for i in range(0, seg_features.shape[1] - chunk_size, chunk_size):
                chunk = seg_features[:, i:i + chunk_size]
                chunk_features.append(np.mean(chunk, axis=1))

            if len(chunk_features) < 2:
                return 1.0

            similarities = []
            for i in range(len(chunk_features)):
                for j in range(i + 1, len(chunk_features)):
                    sim = cosine_similarity([chunk_features[i]], [chunk_features[j]])[0][0]
                    similarities.append(sim)

            return np.mean(similarities) if similarities else 1.0

        except Exception:
            return 1.0

    def _calculate_segment_summary_stats(self, segment_analyses: List[Dict]) -> Dict:
        """
        Calculate summary statistics for individual segment analyses
        """
        confidences = [seg['confidence'] for seg in segment_analyses]

        high_quality = len([s for s in segment_analyses if s['confidence'] > 0.8])
        good_quality = len([s for s in segment_analyses if 0.6 < s['confidence'] <= 0.8])
        fair_quality = len([s for s in segment_analyses if 0.4 < s['confidence'] <= 0.6])
        poor_quality = len([s for s in segment_analyses if s['confidence'] <= 0.4])

        issues_count = {}
        for seg in segment_analyses:
            for issue in seg['issues']:
                issue_type = issue.split('(')[0].strip()
                issues_count[issue_type] = issues_count.get(issue_type, 0) + 1

        return {
            'total_segments': len(segment_analyses),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'quality_distribution': {
                'high_quality': high_quality,
                'good_quality': good_quality,
                'fair_quality': fair_quality,
                'poor_quality': poor_quality
            },
            'common_issues': dict(sorted(issues_count.items(), key=lambda x: x[1], reverse=True)[:5])
        }

    def generate_rule_based_feedback(self, analysis_results: Dict) -> str:
        """
        Generate comprehensive qualitative feedback using rule-based analysis
        """
        feedback_parts = []

        overall_score = analysis_results['overall_confidence']
        if overall_score > 0.8:
            feedback_parts.append("EXCELLENT diarization quality detected.")
        elif overall_score > 0.6:
            feedback_parts.append("GOOD diarization quality with minor issues.")
        elif overall_score > 0.4:
            feedback_parts.append("FAIR diarization quality with several areas for improvement.")
        else:
            feedback_parts.append("POOR diarization quality requiring significant improvements.")

        turn_score = analysis_results['turn_accuracy']['score']
        if turn_score > 0.25:
            feedback_parts.append(
                "✓ Speaker boundaries are well-detected with clear acoustic differences between speakers.")
        elif turn_score > 0.15:
            feedback_parts.append("⚠ Moderate speaker boundary detection - some transitions may be unclear.")
        else:
            feedback_parts.append(
                "✗ Weak speaker boundary detection - many transitions lack clear acoustic differentiation.")

        consistency_score = analysis_results['consistency']['score']
        if consistency_score > 0.7:
            feedback_parts.append("✓ Excellent speaker label consistency throughout the audio.")
        elif consistency_score > 0.5:
            feedback_parts.append("⚠ Good speaker consistency with minor labeling inconsistencies.")
        else:
            feedback_parts.append("✗ Poor speaker consistency - likely speaker confusion or mislabeling.")

        seg_score = analysis_results['segmentation']['score']
        seg_stats = analysis_results['segmentation']['statistics']

        if seg_score > 0.8:
            feedback_parts.append("✓ Well-balanced segmentation with appropriate segment lengths.")
        elif seg_stats['very_short_segments'] > len(self.segments) * 0.2:
            feedback_parts.append(
                "✗ Over-segmentation detected - too many very short segments suggest aggressive splitting.")
        elif seg_stats['very_long_segments'] > 0:
            feedback_parts.append(
                "✗ Under-segmentation detected - very long segments suggest missed speaker transitions.")

        recommendations = []
        if turn_score < 0.2:
            recommendations.append("Consider using better speaker embedding models or VAD preprocessing")
        if consistency_score < 0.6:
            recommendations.append("Implement speaker clustering post-processing to merge similar speakers")
        if seg_stats['rapid_switches'] > len(self.segments) * 0.15:
            recommendations.append("Apply smoothing filters to reduce rapid speaker switching noise")
        if seg_stats['gaps_overlaps'] > len(self.segments) * 0.1:
            recommendations.append("Improve timestamp alignment to reduce gaps and overlaps")

        speaker_counts = {}
        for seg in self.segments:
            speaker_counts[seg['speaker']] = speaker_counts.get(seg['speaker'], 0) + 1

        total_segments = len(self.segments)
        dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])
        if dominant_speaker[1] > total_segments * 0.9:
            recommendations.append(
                f"Warning: {dominant_speaker[0]} dominates {dominant_speaker[1]}/{total_segments} segments - check for under-diarization")

        speaker_names = list(speaker_counts.keys())
        if len(speaker_names) > 1:
            naming_issues = []
            for name in speaker_names:
                if 'SPEAKER_0' in name and len(name.split('_')[-1]) == 1:
                    similar_names = [n for n in speaker_names if n.startswith('SPEAKER_0') and n != name]
                    if similar_names:
                        naming_issues.append(f"Inconsistent speaker naming: {name} vs {similar_names}")

            if naming_issues:
                recommendations.extend(naming_issues)

        if recommendations:
            feedback_parts.append("RECOMMENDATIONS: " + "; ".join(recommendations))

        return " ".join(feedback_parts)

    def run_complete_analysis(self) -> Dict:
        """
        Run complete quality check analysis
        """
        print("=== Starting Complete Quality Check Analysis ===")

        results = {}

        results['turn_accuracy'] = self.analyze_speaker_turn_accuracy()

        results['consistency'] = self.analyze_speaker_consistency()

        results['segmentation'] = self.analyze_segmentation_quality()

        results['individual_analysis'] = self.analyze_individual_segments()

        scores = [
            results['turn_accuracy']['score'],
            results['consistency']['score'],
            results['segmentation']['score']
        ]
        results['overall_confidence'] = np.mean(scores)

        results['detailed_feedback'] = self.generate_rule_based_feedback(results)

        results['summary'] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict) -> str:
        """Generate a comprehensive summary of the analysis"""

        overall_score = results['overall_confidence']

        if overall_score > 0.8:
            quality_level = "Excellent"
        elif overall_score > 0.6:
            quality_level = "Good"
        elif overall_score > 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        summary = f"""
        QUALITY CHECK SUMMARY
        ====================
        Overall Quality: {quality_level} (Score: {overall_score:.2f})

        Component Scores:
        - Speaker Turn Accuracy: {results['turn_accuracy']['score']:.2f}
        - Speaker Consistency: {results['consistency']['score']:.2f}  
        - Segmentation Quality: {results['segmentation']['score']:.2f}

        Key Findings:
        - {results['turn_accuracy']['feedback']}
        - {results['consistency']['feedback']}
        - {results['segmentation']['feedback']}

        Total Segments Analyzed: {len(self.segments)}
        Unique Speakers Detected: {len(set(seg['speaker'] for seg in self.segments))}
        """

        return summary

    def export_results(self, output_file: str, results: Dict):
        """Export detailed results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results exported to {output_file}")
        except Exception as e:
            print(f"Error exporting results: {e}")

    def print_detailed_results(self, results: Dict):
        """Print detailed analysis results"""
        print("\n" + "=" * 60)
        print(results['summary'])
        print("=" * 60)

        print(f"\nDETAILED ANALYSIS:")
        print(f"\n1. SPEAKER TURN ACCURACY:")
        print(f"   Score: {results['turn_accuracy']['score']:.3f}")
        print(f"   {results['turn_accuracy']['feedback']}")
        if 'details' in results['turn_accuracy']:
            print(f"   Transitions analyzed: {len(results['turn_accuracy']['details'])}")

        print(f"\n2. SPEAKER CONSISTENCY:")
        print(f"   Score: {results['consistency']['score']:.3f}")
        print(f"   {results['consistency']['feedback']}")
        if 'speaker_scores' in results['consistency']:
            for speaker, score in results['consistency']['speaker_scores'].items():
                print(f"   {speaker}: {score:.3f}")

        print(f"\n3. SEGMENTATION QUALITY:")
        print(f"   Score: {results['segmentation']['score']:.3f}")
        print(f"   {results['segmentation']['feedback']}")
        if 'statistics' in results['segmentation']:
            stats = results['segmentation']['statistics']
            print(f"   Avg segment duration: {stats['avg_duration']:.2f}s")
            print(f"   Total segments: {stats['total_segments']}")

        print(f"\n4. INDIVIDUAL SEGMENT ANALYSIS:")
        individual_stats = results['individual_analysis']['summary_stats']
        print(f"   Average Segment Confidence: {individual_stats['avg_confidence']:.3f}")
        print(f"   Quality Distribution:")
        dist = individual_stats['quality_distribution']
        print(f"     High Quality: {dist['high_quality']} segments")
        print(f"     Good Quality: {dist['good_quality']} segments")
        print(f"     Fair Quality: {dist['fair_quality']} segments")
        print(f"     Poor Quality: {dist['poor_quality']} segments")

        if individual_stats['common_issues']:
            print(f"   Most Common Issues:")
            for issue, count in list(individual_stats['common_issues'].items())[:3]:
                print(f"     {issue}: {count} segments")

        print(f"\n5. DETAILED FEEDBACK:")
        print(f"   {results['detailed_feedback']}")

        print("\n" + "=" * 60)

    def print_individual_segment_results(self, results: Dict, max_segments: int = 10):
        """Print detailed results for individual segments"""
        individual_segments = results['individual_analysis']['individual_segments']

        print(f"\nINDIVIDUAL SEGMENT ANALYSIS (showing first {min(max_segments, len(individual_segments))} segments):")
        print("=" * 80)

        for i, seg in enumerate(individual_segments[:max_segments]):
            print(f"\nSegment {seg['segment_id']} [{seg['start']:.1f}s - {seg['end']:.1f}s] | "
                  f"Speaker: {seg['speaker']} | Duration: {seg['duration']:.1f}s")
            print(f"  Confidence: {seg['confidence']:.3f}")
            print(f"  Feedback: {seg['feedback']}")
            if seg['text']:
                print(f"  Text: {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}")

        if len(individual_segments) > max_segments:
            print(f"\n... and {len(individual_segments) - max_segments} more segments")
            print(
                "\nTo see all segments, use: agent.print_individual_segment_results(results, max_segments=len(results['individual_analysis']['individual_segments']))")

        print("=" * 80)


if __name__ == "__main__":
    print("Enhanced Quality Check Agent for Speaker Diarization")
    print("=" * 50)

    agent = EnhancedQualityCheckAgent(
        diarization_file='diarization_segments.txt',
        audio_file='inputaudio.wav',
        transcript_file=None
    )

    results = agent.run_complete_analysis()
    agent.print_detailed_results(results)

    agent.export_results('quality_check_results.json', results)