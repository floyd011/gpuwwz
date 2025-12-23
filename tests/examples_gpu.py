"""
This module provides examples demonstrating the WWZ by looking at a simple signal (2 Hz)
with GPU acceleration support.

Modified to use the updated libwwz library with GPU support.

NOTE: The WWZ shows better information on frequency and WWA shows better information on amplitude.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import libwwz
from libwwz import wwt, wwt_gpu, wwt_cpu, GPU_AVAILABLE

# Select Mode...
use_gpu = True  # Set to True to use GPU acceleration
parallel = True

# number of time
ntau = 20  # Creates new time with this many divisions.

# linear
freq_low = 1
freq_high = 5
freq_steps = 0.2  # Resolution of frequency steps
freq_lin = np.arange(freq_low, freq_high + freq_steps, freq_steps)

# octave
freq_target = 2
freq_low_oct = 0.5
freq_high_oct = 6.5
band_order = 3
log_scale_base = 10**(3/10)
override = False

# Create octave frequencies
freq_oct = []
f = freq_target
while f >= freq_low_oct:
    freq_oct.append(f)
    f /= log_scale_base
f = freq_target * log_scale_base
while f <= freq_high_oct:
    freq_oct.append(f)
    f *= log_scale_base
freq_oct = np.sort(np.unique(freq_oct))

# decay constant (c < 0.02) where c = 1/(2*w^2)
# The analyzing wavelet decays significantly in a single cycle 2*pi/w, where w = 2*pi*f
f = 2
w = 2 * np.pi * f
c = 1/(2*w**2)

# Code to remove data points at random
def remove_fraction_with_seed(data, fraction, seed=np.random.randint(1)):
    """
    removes fraction of data at random with given seed.
    :param data: data to remove
    :param fraction: fraction to remove
    :param seed: seed for randomness
    :return: data with fraction removed
    """
    n_to_remove = int(len(data) * fraction)
    np.random.seed(seed)
    return np.delete(data, np.random.choice(np.arange(len(data)), n_to_remove, replace=False))


def run_examples() -> None:
    """
    An example of WWZ/WWA using a sine function time series with missing data will be shown.
    """
    
    print("=" * 60)
    print("WWZ EXAMPLE WITH GPU SUPPORT")
    print("=" * 60)
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"Using GPU: {use_gpu and GPU_AVAILABLE}")
    
    if use_gpu and GPU_AVAILABLE:
        print(f"GPU info:\n{libwwz.gpu_info()}")
        libwwz.set_gpu_device(0)
    
    print("=" * 60)

    # Set timestamps
    sample_freq = 80
    timestamp = np.arange(0, 60, 1 / sample_freq)

    # Create simple signal (2hz)
    sine_2hz = np.sin(timestamp * 2 * (2 * np.pi))
    simple_signal = sine_2hz

    # Remove 80% of the signal at random
    simple_removed = remove_fraction_with_seed(simple_signal, 0.8)
    timestamp_removed = remove_fraction_with_seed(timestamp, 0.8)

    print(f"\nData sizes:")
    print(f"  Full data: {len(timestamp)} points")
    print(f"  Removed data: {len(timestamp_removed)} points")
    print(f"  Linear frequencies: {len(freq_lin)} points")
    print(f"  Octave frequencies: {len(freq_oct)} points")
    
    # Define tau arrays
    tau_lin = np.linspace(timestamp.min(), timestamp.max(), ntau)
    tau_removed = np.linspace(timestamp_removed.min(), timestamp_removed.max(), ntau)

    # Get the WWZ/WWA of the signals (linear)
    # 'linear'
    print("\n" + "=" * 60)
    print("LINEAR METHOD")
    print("=" * 60)
    
    starttime = time.time()
    
    # Choose function based on GPU availability
    if use_gpu and GPU_AVAILABLE:
        wwt_func = wwt_gpu
        gpu_params = {"gpu_chunk_size": 10, "optimize_memory": True}
        print("Using GPU acceleration...")
    else:
        wwt_func = wwt_cpu
        gpu_params = {}
        print("Using CPU only...")
    
    # Full data - linear
    print("Processing full data (linear)...")
    WWZ_simple_linear_result = wwt_func(timestamp, simple_signal, freq_lin, tau=tau_lin, c=c, **gpu_params)
    WWZ_simple_linear = (tau_lin, freq_lin, WWZ_simple_linear_result[0], WWZ_simple_linear_result[1])
    print(f"  Time: {round(time.time() - starttime, 2)} seconds")
    
    # Removed data - linear
    print("Processing removed data (linear)...")
    WWZ_simple_removed_linear_result = wwt_func(timestamp_removed, simple_removed, freq_lin, 
                                                 tau=tau_removed, c=c, **gpu_params)
    WWZ_simple_removed_linear = (tau_removed, freq_lin, 
                                  WWZ_simple_removed_linear_result[0], 
                                  WWZ_simple_removed_linear_result[1])
    print(f"  Time: {round(time.time() - starttime, 2)} seconds")
    
    # 'octave'
    print("\n" + "=" * 60)
    print("OCTAVE METHOD")
    print("=" * 60)
    
    # Full data - octave
    print("Processing full data (octave)...")
    WWZ_simple_octave_result = wwt_func(timestamp, simple_signal, freq_oct, tau=tau_lin, c=c, **gpu_params)
    WWZ_simple_octave = (tau_lin, freq_oct, WWZ_simple_octave_result[0], WWZ_simple_octave_result[1])
    print(f"  Time: {round(time.time() - starttime, 2)} seconds")
    
    # Removed data - octave
    print("Processing removed data (octave)...")
    WWZ_simple_removed_octave_result = wwt_func(timestamp_removed, simple_removed, freq_oct, 
                                                 tau=tau_removed, c=c, **gpu_params)
    WWZ_simple_removed_octave = (tau_removed, freq_oct, 
                                   WWZ_simple_removed_octave_result[0], 
                                   WWZ_simple_removed_octave_result[1])
    total_time = round(time.time() - starttime, 2)
    print(f"  Time: {total_time} seconds")
    
    print(f"\n✅ Total processing time: {total_time} seconds")
    if use_gpu and GPU_AVAILABLE:
        print("🎯 GPU acceleration was used")
    else:
        print("ℹ️  CPU was used (install CuPy for GPU acceleration)")

    # Plot
    plt.rcParams["figure.figsize"] = [14, 6]
    plt.rcParams.update({'font.size': 14})

    # Plot of base functions
    plt.figure(0)
    plt.plot(timestamp, simple_signal, '-', alpha=0.7)
    plt.plot(timestamp_removed, simple_removed, 'o', markersize=3, alpha=0.7)
    plt.ylabel("Amplitude")
    plt.legend(['Full data', '80% removed'], loc='best', fontsize=10)
    plt.xlabel("Time (s)")
    plt.title('Simple 2 Hz Signal with Missing Data')
    plt.grid(True, alpha=0.3)

    # Plot of WWZ for simple and simple removed
    # 'linear'
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    
    # WWZ - Full data
    im1 = ax[0, 0].contourf(WWZ_simple_linear[0], WWZ_simple_linear[1], WWZ_simple_linear[2], 
                           levels=50, cmap='viridis')
    ax[0, 0].set_ylabel('Frequency (Hz)')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([1, 2, 3, 4, 5])
    ax[0, 0].set_title('WWZ - Full Data')
    plt.colorbar(im1, ax=ax[0, 0])
    
    # WWZ - Removed data
    im2 = ax[1, 0].contourf(WWZ_simple_removed_linear[0], WWZ_simple_removed_linear[1], 
                           WWZ_simple_removed_linear[2], levels=50, cmap='viridis')
    ax[1, 0].set_ylabel('Frequency (Hz)')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_yticks([1, 2, 3, 4, 5])
    plt.colorbar(im2, ax=ax[1, 0])
    
    # WWA - Full data
    im3 = ax[0, 1].contourf(WWZ_simple_linear[0], WWZ_simple_linear[1], WWZ_simple_linear[3], 
                           levels=50, cmap='plasma')
    ax[0, 1].set_title('WWA - Full Data')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plt.colorbar(im3, ax=ax[0, 1])
    
    # WWA - Removed data
    im4 = ax[1, 1].contourf(WWZ_simple_removed_linear[0], WWZ_simple_removed_linear[1], 
                           WWZ_simple_removed_linear[3], levels=50, cmap='plasma')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_yticks([])
    plt.colorbar(im4, ax=ax[1, 1])
    
    fig.suptitle(f'Linear Method (Processing time: {total_time}s, GPU: {use_gpu and GPU_AVAILABLE})')
    plt.tight_layout()

    # 'octave' - Simple contour plot since wwz_plot module might not be available
    fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    
    # WWZ - Full data (octave)
    im5 = ax2[0, 0].contourf(WWZ_simple_octave[0], WWZ_simple_octave[1], WWZ_simple_octave[2], 
                            levels=50, cmap='viridis')
    ax2[0, 0].set_ylabel('Frequency (Hz)')
    ax2[0, 0].set_xticks([])
    ax2[0, 0].set_title('WWZ - Full Data (Octave)')
    ax2[0, 0].set_yscale('log')
    plt.colorbar(im5, ax=ax2[0, 0])
    
    # WWZ - Removed data (octave)
    im6 = ax2[1, 0].contourf(WWZ_simple_removed_octave[0], WWZ_simple_removed_octave[1], 
                            WWZ_simple_removed_octave[2], levels=50, cmap='viridis')
    ax2[1, 0].set_ylabel('Frequency (Hz)')
    ax2[1, 0].set_xlabel('Time (s)')
    ax2[1, 0].set_yscale('log')
    plt.colorbar(im6, ax=ax2[1, 0])
    
    # WWA - Full data (octave)
    im7 = ax2[0, 1].contourf(WWZ_simple_octave[0], WWZ_simple_octave[1], WWZ_simple_octave[3], 
                            levels=50, cmap='plasma')
    ax2[0, 1].set_title('WWA - Full Data (Octave)')
    ax2[0, 1].set_xticks([])
    ax2[0, 1].set_yticks([])
    ax2[0, 1].set_yscale('log')
    plt.colorbar(im7, ax=ax2[0, 1])
    
    # WWA - Removed data (octave)
    im8 = ax2[1, 1].contourf(WWZ_simple_removed_octave[0], WWZ_simple_removed_octave[1], 
                            WWZ_simple_removed_octave[3], levels=50, cmap='plasma')
    ax2[1, 1].set_xlabel('Time (s)')
    ax2[1, 1].set_yticks([])
    ax2[1, 1].set_yscale('log')
    plt.colorbar(im8, ax=ax2[1, 1])
    
    fig2.suptitle(f'Octave Method (Processing time: {total_time}s, GPU: {use_gpu and GPU_AVAILABLE})')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_examples()
