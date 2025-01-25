"""
A library for audio rhythm analysis and click track generation.

This module provides the RhythmEngine class which analyzes audio files to detect
beats, generate synchronized click tracks, and export the processed audio.

Example usage:
    >>> engine = RhythmEngine("input_audio.wav")
    >>> mixed_audio = engine.generate_click_track()
    >>> engine.save_mixed_audio("output_with_clicks.wav")
    >>> beat_times = engine.get_beat_times()
    >>> print(f"Detected beats at: {beat_times}")
"""

import librosa
import numpy as np
from multiprocessing import Pool
from pydub import AudioSegment
from typing import Optional, List, Tuple


class RhythmEngine:
    """
    A rhythm processing engine for beat detection and click track generation.

    Attributes:
        audio_path (str): Path to the input audio file
        y (np.ndarray): Audio time series
        sr (int): Sampling rate of the audio
        beat_times (Optional[List[float]]): List of detected beat times in seconds
        mixed_audio (Optional[np.ndarray]): Audio mixed with generated click track
    """

    def __init__(self, audio_path: str) -> None:
        """
        Initialize the rhythm engine with an audio file.

        Args:
            audio_path: Path to the audio file to process

        Raises:
            FileNotFoundError: If the specified audio file doesn't exist
            librosa.LibrosaError: If there's an error loading the audio file
        """
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None, mono=True)
        self.beat_times: Optional[List[float]] = None
        self.mixed_audio: Optional[np.ndarray] = None
        self._process_audio()

    def _process_chunk(self, args: Tuple[int, np.ndarray, float, int, int]) -> np.ndarray:
        """
        Process a chunk of audio data for beat tracking (internal helper).

        Args:
            args: Tuple containing:
                - chunk_idx: Index of the chunk
                - onset_chunk: Onset envelope segment for this chunk
                - tempo: Estimated global tempo
                - start_frame: Starting frame index for this chunk
                - hop_length: Hop length used in analysis

        Returns:
            np.ndarray: Beat frames detected in this chunk
        """
        chunk_idx, onset_chunk, tempo, start_frame, hop_length = args
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_chunk,
            sr=self.sr,
            start_bpm=tempo,
            hop_length=hop_length,
            tightness=100
        )
        return start_frame + beat_frames

    def _process_audio(self) -> None:
        """Analyze audio to detect beats using parallel processing."""
        hop_length = 512
        onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr,
            hop_length=hop_length,
            aggregate=np.median,
            n_fft=2048
        )

        # Estimate global tempo
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=hop_length,
            units='frames'
        )

        # Split analysis into parallel chunks
        total_frames = len(onset_env)
        n_chunks = min(4, max(1, total_frames // 100))
        onset_chunks = np.array_split(onset_env, n_chunks)

        start_frames = [0]
        for chunk in onset_chunks[:-1]:
            start_frames.append(start_frames[-1] + len(chunk))

        args_list = [
            (i, chunk, tempo, start_frames[i], hop_length)
            for i, chunk in enumerate(onset_chunks)
            if len(chunk) > 0
        ]

        with Pool() as pool:
            results = pool.map(self._process_chunk, args_list)

        beat_frames = np.concatenate(results) if results else np.array([])
        self.beat_times = librosa.frames_to_time(
            beat_frames,
            sr=self.sr,
            hop_length=hop_length
        ).tolist()
        self.beat_times.sort()

    def generate_click_track(self, click_duration: float = 0.07,
                             freq1: float = 880.0, freq2: float = 1320.0) -> np.ndarray:
        """
        Generate a click track synchronized with detected beats.

        Args:
            click_duration: Duration of each click sound in seconds (default: 0.07)
            freq1: Primary frequency component of the click sound in Hz (default: 880)
            freq2: Secondary frequency component of the click sound in Hz (default: 1320)

        Returns:
            np.ndarray: Mixed audio with click track, normalized to prevent clipping
        """
        click_track = np.zeros_like(self.y)
        t_click = np.linspace(0, click_duration, int(click_duration * self.sr), endpoint=False)
        click_sound = 0.5 * (
            np.sin(2 * np.pi * freq1 * t_click) * np.exp(-3 * t_click) +
            np.sin(2 * np.pi * freq2 * t_click) * np.exp(-4 * t_click)
        )

        onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr,
            hop_length=512,
            aggregate=np.median
        )

        for beat_time in self.beat_times:
            sample_idx = int(beat_time * self.sr)
            if sample_idx >= len(self.y):
                continue

            end_idx = sample_idx + len(click_sound)
            if end_idx > len(self.y):
                end_idx = len(self.y)

            valid_length = end_idx - sample_idx
            if valid_length <= 0:
                continue

            # Calculate dynamic volume based on onset strength
            frame_idx = librosa.time_to_frames(
                beat_time,
                sr=self.sr,
                hop_length=512
            )
            frame_idx = min(max(frame_idx, 0), len(onset_env)-1)
            strength = (onset_env[frame_idx] - onset_env.min()) / (np.ptp(onset_env) + 1e-9)

            # Apply volume envelope with decay
            click_track[sample_idx:end_idx] += (
                click_sound[:valid_length] *
                (0.4 + 0.6 * strength) *
                np.linspace(1, 0.5, valid_length)
            )

        self.mixed_audio = self.y + click_track

        # Normalize to prevent clipping
        peak = np.max(np.abs(self.mixed_audio))
        if peak > 1.0:
            self.mixed_audio /= peak * 1.05

        return self.mixed_audio

    def get_beat_times(self) -> List[float]:
        """
        Get the detected beat times.

        Returns:
            List[float]: List of beat times in seconds, sorted chronologically
        """
        return self.beat_times.copy() if self.beat_times else []

    def save_mixed_audio(self, output_path: str) -> None:
        """
        Save the mixed audio with click track to a file.

        Args:
            output_path: Path to save the output audio file. Format is determined by
                        the file extension (supports any format supported by pydub)

        Raises:
            ValueError: If mixed_audio hasn't been generated yet
        """
        if self.mixed_audio is None:
            raise ValueError("No mixed audio available. Call generate_click_track() first.")

        audio_segment = AudioSegment(
            (self.mixed_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=self.sr,
            sample_width=2,
            channels=1
        )
        audio_segment.export(output_path, format="wav")

#BTW, WHY YOU WATCHING IN THE CODE?!
