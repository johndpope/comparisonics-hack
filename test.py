import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile

# Function to partition the signal into segments
def partition_signal(signal, sample_rate, segment_duration):
    segment_length = int(sample_rate * segment_duration)
    num_segments = len(signal) // segment_length
    return np.array_split(signal[:num_segments * segment_length], num_segments)

# Function to extract frequency-dependent information using FFT
def extract_frequency_info(segment, sample_rate):
    freqs = fft(segment)
    freqs = np.abs(freqs[:len(freqs)//2])
    dominant_freq = np.argmax(freqs) * sample_rate / len(segment)
    amplitude_variance = np.var(segment)
    return dominant_freq, amplitude_variance

# Function to assign colors based on dominant frequency
def assign_color(dominant_freq, freq_ranges, colors):
    for i, freq_range in enumerate(freq_ranges):
        if dominant_freq >= freq_range[0] and dominant_freq <= freq_range[1]:
            return colors[i]
    return colors[-1]

# Main function to process the signal and display the waveform
def display_waveform_with_coloring(wav_file, segment_duration=0.2):
    sample_rate, signal = wavfile.read(wav_file)
    signal = signal / np.max(np.abs(signal))  # Normalizing the signal
    segments = partition_signal(signal, sample_rate, segment_duration)
    
    # Define frequency ranges and corresponding colors from the image
    freq_ranges = [(0, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, sample_rate/2)]
    colors = ['#C7F464', '#61C0BF', '#FDBF57', '#D9A9F2', '#F57188', '#FF4E50']
    
    time_axis = np.linspace(0, len(signal) / sample_rate, len(signal))
    plt.figure(figsize=(15, 5))

    for i, segment in enumerate(segments):
        dominant_freq, _ = extract_frequency_info(segment, sample_rate)
        color = assign_color(dominant_freq, freq_ranges, colors)
        segment_time_axis = time_axis[i * len(segment):(i + 1) * len(segment)]
        plt.plot(segment_time_axis, segment, color=color)
    
    plt.title('Waveform Display with Frequency-Based Coloring')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

# Run the display function on a sample WAV file
display_waveform_with_coloring('mira3.wav')
