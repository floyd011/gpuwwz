"""
Plotting utilities for WWZ results with GPU compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def safe_convert_to_numpy(array):
    """Bezbedno konvertuj u NumPy."""
    if isinstance(array, np.ndarray):
        return array
    try:
        if hasattr(array, 'get'):
            return array.get()
    except:
        pass
    return np.asarray(array)

def plot_wwz_results(t, y, freq, tau, wwz, dampp, title="WWZ Analysis", 
                    use_gpu=False, elapsed_time=None, figsize=(14, 8)):
    """
    Kompletan plot za WWZ rezultate.
    
    Args:
        use_gpu: Da li je korišćen GPU
        elapsed_time: Vreme izvršenja
    """
    # Konvertuj sve u NumPy
    t = safe_convert_to_numpy(t)
    y = safe_convert_to_numpy(y)
    freq = safe_convert_to_numpy(freq)
    tau = safe_convert_to_numpy(tau)
    wwz = safe_convert_to_numpy(wwz)
    dampp = safe_convert_to_numpy(dampp)
    
    # Proveri dimenzije
    if wwz.shape == (len(freq), len(tau)) and wwz.shape != (len(tau), len(freq)):
        wwz = wwz.T  # Transponuj ako treba
        dampp = dampp.T
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Original signal
    axes[0, 0].scatter(t, y, s=1, alpha=0.5, color='blue')
    axes[0, 0].set_title(f'Original Signal\\n{len(t):,} points')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[0, 1].hist(y, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 1].set_title('Amplitude Distribution')
    axes[0, 1].set_xlabel('Amplitude')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Frequencies
    axes[0, 2].plot(freq, 'g-', linewidth=2)
    axes[0, 2].set_title(f'Analysis Frequencies\\n{len(freq)} points')
    axes[0, 2].set_xlabel('Index')
    axes[0, 2].set_ylabel('Frequency (Hz)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. WWZ Power
    im1 = axes[1, 0].imshow(wwz, aspect='auto', origin='lower',
                           extent=[tau.min(), tau.max(), freq.min(), freq.max()],
                           cmap='viridis')
    axes[1, 0].set_title(f'WWZ Power\\nGPU: {use_gpu}' + 
                        (f', Time: {elapsed_time:.1f}s' if elapsed_time else ''))
    axes[1, 0].set_xlabel('Time (tau)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[1, 0], label='WWZ Power')
    
    # 5. Damping
    im2 = axes[1, 1].imshow(dampp, aspect='auto', origin='lower',
                           extent=[tau.min(), tau.max(), freq.min(), freq.max()],
                           cmap='plasma')
    axes[1, 1].set_title('Damping Coefficient')
    axes[1, 1].set_xlabel('Time (tau)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[1, 1], label='Damping')
    
    # 6. Statistics
    axes[1, 2].axis('off')
    
    stats_text = (
        f"⚙️ CONFIGURATION:\\n"
        f"-----------------\\n"
        f"• GPU used: {use_gpu}\\n"
        f"• Time points: {len(t):,}\\n"
        f"• Tau points: {len(tau):,}\\n"
        f"• Frequencies: {len(freq):,}\\n"
        f"• Result shape: {wwz.shape}\\n"
        f"• WWZ range: [{wwz.min():.3f}, {wwz.max():.3f}]\\n"
        f"• Damping range: [{dampp.min():.3f}, {dampp.max():.3f}]"
    )
    
    if elapsed_time:
        stats_text += f"\\n• Execution time: {elapsed_time:.2f}s"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                   verticalalignment='center', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_multi_gpu_comparison(results_dict, figsize=(12, 8)):
    """
    Plot comparison of multi-GPU vs single GPU results.
    
    Args:
        results_dict: Dict with keys like 'single_gpu', 'multi_gpu'
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for idx, (name, (wwz, dampp)) in enumerate(results_dict.items()):
        row = idx // 2
        col = idx % 2
        
        if row >= 2 or col >= 2:
            break
            
        wwz_np = safe_convert_to_numpy(wwz)
        im = axes[row, col].imshow(wwz_np.T, aspect='auto', origin='lower', cmap='viridis')
        axes[row, col].set_title(f'{name}\\nShape: {wwz_np.shape}')
        axes[row, col].set_xlabel('Tau')
        axes[row, col].set_ylabel('Frequency')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    return fig
"""
This module provides functions for plotting the outcomes of wwz.py. It focuses on making proper grids for pcolormesh.

UPDATED FOR GPU-ENABLED LIBWWZ VERSION
"""

import matplotlib.ticker as ticker
import matplotlib.axes as axes
import numpy as np


def safe_convert_to_numpy(array):
    """
    Bezbedno konvertuj bilo koji array u NumPy.
    Radi sa: NumPy array, CuPy array, liste, tuple.
    """
    # Ako je već NumPy array
    if isinstance(array, np.ndarray):
        return array
    
    # Ako je CuPy array
    try:
        if hasattr(array, 'get'):
            return array.get()
    except:
        pass
    
    # Ako je nešto drugo, pokušaj da konvertuješ
    try:
        return np.asarray(array)
    except:
        raise ValueError(f"Cannot convert {type(array)} to NumPy array")


def make_linear_freq_plot_grid(freq_mat: np.ndarray) -> np.ndarray:
    """
    Used for linear method.
    Takes the FREQ output from wwz.py and creates a grid for pcolormesh.
    
    UPDATED: Handles both 1D and 2D freq arrays
    """
    # Konvertuj u NumPy ako nije već
    freq_mat = safe_convert_to_numpy(freq_mat)
    
    # Ako je 1D array (nova verzija), konvertuj u 2D
    if freq_mat.ndim == 1:
        freq_mat = freq_mat.reshape(1, -1)
    
    # Get the array of center frequencies from the freq_mat
    freq_centers = freq_mat[0, :]

    # Get the freq_steps by subtracting the first two freq_centers
    freq_step = freq_centers[1] - freq_centers[0]

    # Subtract half of the freq_step from the freq_centers to get lower bound
    freq_lows = freq_centers - freq_step / 2

    # Append the high frequency bound to get all the boundaries
    freq_highest = freq_centers.max() + freq_step / 2
    freq_bounds = np.append(freq_lows, freq_highest)

    # Tile the freq_bounds to create a grid
    freq_grid = np.tile(freq_bounds, (freq_mat.shape[0] + 1, 1))

    return freq_grid


def make_octave_freq_plot_grid(freq_mat: np.ndarray, band_order: float, log_scale_base: float) -> np.ndarray:
    """
    Used for octave method.
    Takes the FREQ output from wwz.py and creates a grid for pcolormesh
    
    UPDATED: Handles both 1D and 2D freq arrays
    """
    # Konvertuj u NumPy ako nije već
    freq_mat = safe_convert_to_numpy(freq_mat)
    
    # Ako je 1D array (nova verzija), konvertuj u 2D
    if freq_mat.ndim == 1:
        freq_mat = freq_mat.reshape(1, -1)

    # Get the array of the center frequencies from the freq_mat
    freq_centers = freq_mat[0, :]

    # Convert the center frequencies to low frequencies
    freq_lows = freq_centers / log_scale_base**(1 / (2 * band_order))

    # Append the high frequency at the end to get all the boundaries
    freq_highest = freq_centers.max() * log_scale_base**(1 / (2 * band_order))
    freq_bounds = np.append(freq_lows, freq_highest)

    # Tile the freq_bounds to create a grid
    freq_grid = np.tile(freq_bounds, (freq_mat.shape[0] + 1, 1))

    return freq_grid


def make_tau_plot_grid(tau_mat: np.ndarray) -> np.ndarray:
    """
    Used for both octave and linear.
    Takes the TAU output from wwz.py and creates a grid for pcolormesh
    
    UPDATED: Handles both 1D and 2D tau arrays
    """
    # Konvertuj u NumPy ako nije već
    tau_mat = safe_convert_to_numpy(tau_mat)
    
    # Ako je 1D array (nova verzija), konvertuj u 2D
    if tau_mat.ndim == 1:
        tau_mat = tau_mat.reshape(-1, 1)

    # Get the tau values from tau_mat
    taus = tau_mat[:, 0]

    # Append one tau value for edge limit by adding the step to the largest tau
    taus = np.append(taus, taus[-1] + taus[1] - taus[0])

    # Tile the taus with an additional column to create grid that matches freq_grid
    tau_grid = np.tile(taus, (tau_mat.shape[1] + 1, 1)).transpose()

    return tau_grid


def prepare_wwz_data(wwz_data, transpose_for_plot=True):
    """
    Priprema WWZ podatke za plotovanje.
    
    Nova wwt vraća (nfreq × ntau)
    Stari plotter očekuje (ntau × nfreq)
    
    Args:
        wwz_data: WWZ rezultati (može biti 2D array ili tuple)
        transpose_for_plot: Da li treba transponovati za plot
    
    Returns:
        Properly formatted data for plotting
    """
    wwz_data = safe_convert_to_numpy(wwz_data)
    
    # Ako je tuple (wwz, dampp), obradi svaki element
    if isinstance(wwz_data, tuple) and len(wwz_data) == 2:
        wwz, dampp = wwz_data
        wwz = safe_convert_to_numpy(wwz)
        dampp = safe_convert_to_numpy(dampp)
        
        if transpose_for_plot:
            return wwz.T, dampp.T
        else:
            return wwz, dampp
    
    # Ako je samo jedan array
    if transpose_for_plot and wwz_data.ndim == 2:
        return wwz_data.T
    else:
        return wwz_data


def linear_plotter(ax: axes, TAU, FREQ, DATA, **kwargs):
    """
    Creates a plot for the 'linear' method.
    UPDATED: Compatible with both old and new wwt outputs
    """
    # Konvertuj sve u NumPy
    TAU = safe_convert_to_numpy(TAU)
    FREQ = safe_convert_to_numpy(FREQ)
    DATA = safe_convert_to_numpy(DATA)
    
    # Proveri dimenzije i prilagodi ako je potrebno
    # Nova wwt vraća DATA kao (nfreq × ntau), a plotter očekuje (ntau × nfreq)
    if DATA.shape == (FREQ.size, TAU.size) and DATA.shape != (TAU.size, FREQ.size):
        DATA = DATA.T  # Transponuj
    
    # Create grid for pcolormesh boundaries
    tau_grid = make_tau_plot_grid(TAU)
    freq_grid = make_linear_freq_plot_grid(FREQ)
    
    # Proveri da li se gridovi slažu sa DATA dimenzijama
    # pcolormesh očekuje grid dimenzije: (tau_grid: ntau+1 × nfreq+1, freq_grid: ntau+1 × nfreq+1)
    # DATA dimenzije: (ntau × nfreq)
    
    # Plot using subplots
    im = ax.pcolormesh(tau_grid, freq_grid, DATA, **kwargs)

    # Add color bar and fix y_ticks
    ax.figure.colorbar(im, ax=ax)
    
    # Postavi y ticks samo ako nisu previše gusti
    if FREQ.size <= 20:  # Ako ima manje od 20 frekvencija
        ax.set_yticks(FREQ.ravel() if FREQ.ndim > 1 else FREQ)
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    
    return im


def octave_plotter(ax: axes, TAU, FREQ, DATA,
                   band_order: float, log_scale_base: float, 
                   log_y_scale: bool = True, **kwargs):
    """
    Creates a plot for the 'octave' method.
    UPDATED: Compatible with both old and new wwt outputs
    """
    # Konvertuj sve u NumPy
    TAU = safe_convert_to_numpy(TAU)
    FREQ = safe_convert_to_numpy(FREQ)
    DATA = safe_convert_to_numpy(DATA)
    
    # Proveri dimenzije i prilagodi ako je potrebno
    if DATA.shape == (FREQ.size, TAU.size) and DATA.shape != (TAU.size, FREQ.size):
        DATA = DATA.T  # Transponuj
    
    # Create grid for pcolormesh boundaries
    tau_grid = make_tau_plot_grid(TAU)
    freq_grid = make_octave_freq_plot_grid(FREQ, band_order, log_scale_base)

    # Plot using subplots
    im = ax.pcolormesh(tau_grid, freq_grid, DATA, **kwargs)

    # Add color bar, fix y_scale, and fix y_ticks
    ax.figure.colorbar(im, ax=ax)
    if log_y_scale is True:
        ax.set_yscale('log', base=log_scale_base)
    
    # Postavi y ticks
    if FREQ.size <= 20:  # Ako ima manje od 20 frekvencija
        ax.set_yticks(FREQ.ravel() if FREQ.ndim > 1 else FREQ)
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    
    return im


def plot_wwz_comprehensive(TAU, FREQ, WWZ, WWA=None, 
                          method='linear', figsize=(14, 6), 
                          titles=None, **kwargs):
    """
    Kompletan plot za WWZ i WWA rezultate.
    
    Args:
        TAU: Vreme/tau tačke (1D ili 2D)
        FREQ: Frekvencije (1D ili 2D)
        WWZ: WWZ power rezultati
        WWA: WWA amplitude rezultati (opciono)
        method: 'linear' ili 'octave'
        figsize: Veličina figure
        titles: Lista naslova [wwz_title, wwa_title]
        **kwargs: Dodatni parametri za octave_plotter
    """
    # Podrazumevani naslovi
    if titles is None:
        titles = ['WWZ Power', 'WWA Amplitude']
    
    # Kreiraj figure
    if WWA is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        axes = [ax]
    
    # Plot WWZ
    if method == 'linear':
        im1 = linear_plotter(axes[0], TAU, FREQ, WWZ)
    else:  # octave
        im1 = octave_plotter(axes[0], TAU, FREQ, WWZ, **kwargs)
    
    axes[0].set_title(titles[0])
    
    # Plot WWA ako postoji
    if WWA is not None and len(axes) > 1:
        if method == 'linear':
            im2 = linear_plotter(axes[1], TAU, FREQ, WWA)
        else:  # octave
            im2 = octave_plotter(axes[1], TAU, FREQ, WWA, **kwargs)
        
        axes[1].set_title(titles[1])
    
    plt.tight_layout()
    return fig


# Nova funkcija specijalno za novu wwt_memory_safe
def plot_modern_wwz(t, y, freq=None, tau=None, use_gpu=False, 
                    method='linear', **kwargs):
    """
    Jednostavna funkcija koja radi WWZ i plotuje rezultate.
    Koristi novu wwt_memory_safe funkciju.
    
    Args:
        t: Vremenske tačke
        y: Vrednosti
        freq: Frekvencije (ako je None, automatski generiše)
        tau: Tau tačke (ako je None, koristi t)
        use_gpu: Da li koristiti GPU
        method: 'linear' ili 'octave'
    """
    import libwwz
    
    # Generiši frekvencije ako nisu date
    if freq is None:
        t_range = t.max() - t.min()
        if t_range > 0:
            freq_low = 0.1 / t_range
        else:
            freq_low = 0.01
        freq = np.linspace(freq_low, 2.0, 50)
    
    # Koristi tau ako je dato, inače koristi vreme
    if tau is None:
        tau = np.linspace(t.min(), t.max(), min(100, len(t)))
    
    # Pokreni WWZ
    wwz, dampp = libwwz.wwt_memory_safe(
        t, y, freq, tau=tau, 
        use_gpu=use_gpu, 
        max_memory_gb=6.0
    )
    
    # Plotuj
    fig = plot_wwz_comprehensive(
        tau, freq, wwz.T, dampp.T,  # Transponuj za plot
        method=method,
        titles=[f'WWZ Power (GPU: {use_gpu})', f'Damping (GPU: {use_gpu})'],
        **kwargs
    )
    
    return fig, (wwz, dampp)


if __name__ == "__main__":
    # Test funkcija
    import matplotlib.pyplot as plt
    
    print("Testing plot_methods with GPU compatibility...")
    
    # Generiši test podatke
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    y = np.sin(2*np.pi*0.1*t) + 0.3*np.random.randn(len(t))
    freq = np.linspace(0.01, 0.5, 50)
    tau = np.linspace(t.min(), t.max(), 100)
    
    # Testiraj sa novom funkcijom (ako je libwwz instaliran)
    try:
        import libwwz
        print("libwwz found, testing plot_modern_wwz...")
        
        fig, results = plot_modern_wwz(t, y, freq, tau, use_gpu=False)
        plt.savefig('test_plot_modern.png', dpi=150)
        print("Test plot saved as 'test_plot_modern.png'")
        plt.show()
        
    except ImportError:
        print("libwwz not installed, skipping advanced test")
        
        # Test samo plot funkcije sa random podacima
        wwz_test = np.random.randn(len(tau), len(freq))
        wwa_test = np.random.randn(len(tau), len(freq))
        
        fig = plot_wwz_comprehensive(
            tau, freq, wwz_test, wwa_test,
            method='linear',
            titles=['Test WWZ', 'Test WWA']
        )
        
        plt.savefig('test_plot_basic.png', dpi=150)
        print("Basic test plot saved as 'test_plot_basic.png'")
        plt.show()
