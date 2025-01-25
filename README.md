# RhythmEngine 🎵

A Python library for audio rhythm analysis and click track generation. Detect beats in audio files and generate synchronized click tracks with dynamic volume levels.

## Features

- **Beat Detection**: Multi-processed analysis for accurate beat tracking
- **Click Track Generation**: Create dynamic click sounds that sync with detected beats
- **Audio Export**: Save processed audio with click tracks in standard formats
- **Customizable Parameters**: Adjust click sounds and analysis parameters

## Installation

### Requirements
- Python 3.8+
- FFmpeg (for audio file support)

### Dependencies
```bash
pip install librosa numpy pydub
```

### Recommended Installation
```bash
# Clone repository
git clone https://github.com/yourusername/RhythmEngine.git
cd RhythmEngine

# Install requirements
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from RhythmEngine import RhythmEngine

# Initialize with audio file
engine = RhythmEngine("input_song.mp3")

# Generate click track
mixed_audio = engine.generate_click_track()

# Save output
engine.save_mixed_audio("output_with_clicks.wav")

# Get detected beats
beat_times = engine.get_beat_times()
print(f"Detected beats at: {beat_times}")
```

### Advanced Usage
```python
# Custom click sound parameters
engine.generate_click_track(
    click_duration=0.1,  # Click duration in seconds
    freq1=1000,          # Primary frequency (Hz)
    freq2=1500           # Secondary frequency (Hz)
)

# Access raw audio data
original_audio = engine.y          # Raw audio waveform
sample_rate = engine.sr            # Sampling rate
mixed_audio = engine.mixed_audio   # Processed audio with clicks
```

## API Documentation

### `RhythmEngine(audio_path: str)`
Main class for rhythm processing

**Parameters**:
- `audio_path`: Path to audio file (supports most common formats)

### Methods
#### `generate_click_track(click_duration=0.07, freq1=880, freq2=1320)`
Generate click track synchronized with detected beats

**Parameters**:
- `click_duration`: Duration of click sound in seconds (default: 0.07)
- `freq1`: Primary frequency component in Hz (default: 880)
- `freq2`: Secondary frequency component in Hz (default: 1320)

**Returns**:
- `np.ndarray`: Mixed audio waveform with click track

#### `save_mixed_audio(output_path: str)`
Save processed audio to file

**Parameters**:
- `output_path`: Output file path (format determined by extension)

#### `get_beat_times()`
Get detected beat times

**Returns**:
- `List[float]`: List of beat times in seconds

## Implementation Details

- **Multi-processing**: Uses parallel processing for faster beat detection
- **Dynamic Volume**: Click volume adapts to onset strength of original audio
- **Normalization**: Automatic peak normalization prevents clipping
- **Tempo Estimation**: Combines global tempo estimation with local beat tracking

## Supported Formats
Input: MP3, WAV, FLAC, OGG, AAC (any format supported by FFmpeg)  
Output: WAV (other formats supported via Pydub)

## Acknowledgments
- Uses [librosa](https://librosa.org/) for audio analysis
- Audio processing with [pydub](https://github.com/jiaaro/pydub)
- Built with NumPy for efficient array operations

---

**Contributions welcome!** Please open an issue or PR for suggestions/bug reports.
