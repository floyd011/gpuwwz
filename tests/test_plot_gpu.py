# test_plot_gpu.py
import numpy as np
import matplotlib.pyplot as plt
import libwwz

print("Testing GPU-enabled libwwz with plotting...")
print(f"GPU available: {libwwz.GPU_AVAILABLE}")

# Generiši test podatke
np.random.seed(42)
t = np.linspace(0, 100, 2000)
y = np.sin(2*np.pi*0.1*t) + 0.5*np.sin(2*np.pi*0.3*t) + 0.2*np.random.randn(len(t))
freq = np.linspace(0.01, 0.5, 100)
tau = np.linspace(t.min(), t.max(), 200)

print(f"Data: {len(t)} time points, {len(freq)} frequencies, {len(tau)} tau points")

# Test 1: Memory Safe WWZ sa GPU
print("\n1. Running Memory Safe WWZ with GPU...")
wwz_gpu, damp_gpu = libwwz.wwt_memory_safe(
    t, y, freq, tau=tau,
    use_gpu=True,
    max_memory_gb=6.0
)
print(f"   GPU results shape: {wwz_gpu.shape}")

# Test 2: Memory Safe WWZ sa CPU
print("\n2. Running Memory Safe WWZ with CPU...")
wwz_cpu, damp_cpu = libwwz.wwt_memory_safe(
    t, y, freq, tau=tau,
    use_gpu=False,
    max_memory_gb=6.0
)
print(f"   CPU results shape: {wwz_cpu.shape}")

# Test 3: Plot sa originalnim plot_methods
print("\n3. Plotting with plot_methods...")
try:
    # Koristi helper funkciju za pripremu podataka
    TAU_grid = tau.reshape(-1, 1)
    FREQ_grid = freq.reshape(1, -1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot GPU rezultati
    libwwz.linear_plotter(axes[0, 0], TAU_grid, FREQ_grid, wwz_gpu.T)
    axes[0, 0].set_title('WWZ Power (GPU)')
    
    libwwz.linear_plotter(axes[0, 1], TAU_grid, FREQ_grid, damp_gpu.T)
    axes[0, 1].set_title('Damping (GPU)')
    
    # Plot CPU rezultati
    libwwz.linear_plotter(axes[1, 0], TAU_grid, FREQ_grid, wwz_cpu.T)
    axes[1, 0].set_title('WWZ Power (CPU)')
    
    libwwz.linear_plotter(axes[1, 1], TAU_grid, FREQ_grid, damp_cpu.T)
    axes[1, 1].set_title('Damping (CPU)')
    
    plt.tight_layout()
    plt.savefig('gpu_vs_cpu_comparison.png', dpi=150)
    print("   Plot saved as 'gpu_vs_cpu_comparison.png'")
    plt.show()
    
except Exception as e:
    print(f"   Plot error: {e}")
    
    # Fallback na jednostavan plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(wwz_gpu.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('WWZ GPU')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(damp_gpu.T, aspect='auto', origin='lower', cmap='plasma')
    plt.title('Damping GPU')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(wwz_cpu.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('WWZ CPU')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(damp_cpu.T, aspect='auto', origin='lower', cmap='plasma')
    plt.title('Damping CPU')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('simple_comparison.png', dpi=150)
    print("   Simple plot saved as 'simple_comparison.png'")
    plt.show()

print("\n✅ All tests completed!")
