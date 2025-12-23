# test_multi_cpu.py
import numpy as np
import libwwz
import time

print("="*60)
print("🖥️ MULTI-CPU WWZ TEST")
print("="*60)

# System info
print(f"📊 System information:")
print(f"libwwz version: {libwwz.__version__}")
print(f"CPU cores: {libwwz.get_cpu_count()}")
print(f"GPU available: {libwwz.GPU_AVAILABLE}")

# Generate test data
print("\\n📈 Generating test data...")
np.random.seed(42)

# Veći dataset za bolji test
t = np.sort(np.random.uniform(0, 100, 20000))
y = np.sin(2*np.pi*0.1*t) + 0.5*np.sin(2*np.pi*0.3*t) + 0.2*np.random.randn(len(t))
freq = np.linspace(0.01, 0.5, 300)
tau = np.linspace(t.min(), t.max(), 200)

print(f"Time points: {len(t):,}")
print(f"Frequencies: {len(freq):,}")
print(f"Tau points: {len(tau):,}")

# Test 1: Single CPU
print("\\n1. 🎯 SINGLE CPU TEST")
start = time.time()
wwz_single, damp_single = libwwz.wwt_memory_safe(
    t, y, freq, tau=tau,
    use_gpu=False,
    multi_cpu=False
)
single_time = time.time() - start
print(f"   ✅ Completed in {single_time:.2f}s")
print(f"   Shape: {wwz_single.shape}")

# Test 2: Multi-CPU (sve jezgre)
print("\\n2. 🚀 MULTI-CPU TEST (all cores)")
start = time.time()
wwz_multi, damp_multi = libwwz.wwt_memory_safe(
    t, y, freq, tau=tau,
    use_gpu=False,
    multi_cpu=True,
    n_cpu_workers=None  # Auto-detect
)
multi_time = time.time() - start
print(f"   ✅ Completed in {multi_time:.2f}s")
print(f"   Shape: {wwz_multi.shape}")

# Calculate speedup
if single_time > 0 and multi_time > 0:
    speedup = single_time / multi_time
    print(f"   ⚡ Speedup: {speedup:.1f}x")
    
    efficiency = (speedup / libwwz.get_cpu_count()) * 100
    print(f"   📊 Efficiency: {efficiency:.0f}% of ideal")

# Test 3: Multi-CPU sa određenim brojem workers
print("\\n3. 🔧 MULTI-CPU TEST (4 workers)")
start = time.time()
wwz_4core, damp_4core = libwwz.wwt_memory_safe(
    t, y, freq, tau=tau,
    use_gpu=False,
    multi_cpu=True,
    n_cpu_workers=31
)
fourcore_time = time.time() - start
print(f"   ✅ Completed in {fourcore_time:.2f}s")

# Verify results
print("\\n4. 📏 VERIFYING RESULTS")
max_diff_single_multi = np.max(np.abs(wwz_single - wwz_multi))
max_diff_single_4core = np.max(np.abs(wwz_single - wwz_4core))

print(f"   Single vs Multi-CPU max difference: {max_diff_single_multi:.2e}")
print(f"   Single 31-core max difference: {max_diff_single_4core:.2e}")

if max_diff_single_multi < 1e-5 and max_diff_single_4core < 1e-5:
    print("   ✅ All results match within tolerance")
else:
    print("   ⚠️ Small differences detected")

# Performance summary
print("\\n5. 📈 PERFORMANCE SUMMARY")
print("-"*40)

print(f"{'Method':<20} {'Time (s)':<12} {'Speedup':<10}")
print(f"{'-'*20} {'-'*12} {'-'*10}")
print(f"{'Single CPU':<20} {single_time:<12.2f} {'1.0x':<10}")
print(f"{'Multi-CPU (all)':<20} {multi_time:<12.2f} {single_time/multi_time:<10.1f}x")
print(f"{'Multi-CPU (31 cores)':<20} {fourcore_time:<12.2f} {single_time/fourcore_time:<10.1f}x")

# Plot results
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Single CPU
    axes[0, 0].imshow(wwz_single.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Single CPU\\n{single_time:.1f}s')
    axes[0, 0].set_xlabel('Tau')
    axes[0, 0].set_ylabel('Frequency')
    
    # Multi-CPU all cores
    axes[0, 1].imshow(wwz_multi.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title(f'Multi-CPU (all)\\n{multi_time:.1f}s')
    axes[0, 1].set_xlabel('Tau')
    axes[0, 1].set_ylabel('Frequency')
    
    # 4-core
    axes[0, 2].imshow(wwz_4core.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 2].set_title(f'Multi-CPU (4 cores)\\n{fourcore_time:.1f}s')
    axes[0, 2].set_xlabel('Tau')
    axes[0, 2].set_ylabel('Frequency')
    
    # Difference plots
    diff1 = np.abs(wwz_single - wwz_multi)
    im1 = axes[1, 0].imshow(diff1.T, aspect='auto', origin='lower', cmap='hot')
    axes[1, 0].set_title(f'Single vs Multi diff\\nmax: {diff1.max():.2e}')
    axes[1, 0].set_xlabel('Tau')
    axes[1, 0].set_ylabel('Frequency')
    plt.colorbar(im1, ax=axes[1, 0])
    
    diff2 = np.abs(wwz_single - wwz_4core)
    im2 = axes[1, 1].imshow(diff2.T, aspect='auto', origin='lower', cmap='hot')
    axes[1, 1].set_title(f'Single vs 31-core diff\\nmax: {diff2.max():.2e}')
    axes[1, 1].set_xlabel('Tau')
    axes[1, 1].set_ylabel('Frequency')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Performance bar chart
    methods = ['Single CPU', 'Multi-CPU (all)', 'Multi-CPU (31)']
    times = [single_time, multi_time, fourcore_time]
    colors = ['blue', 'green', 'orange']
    
    axes[1, 2].bar(methods, times, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel('Time (s)')
    axes[1, 2].set_title('Performance Comparison')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add time values on bars
    for i, (method, time_val) in enumerate(zip(methods, times)):
        axes[1, 2].text(i, time_val + 0.1, f'{time_val:.1f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig('multi_cpu_comparison.png', dpi=150)
    print("\\n📊 Plot saved as 'multi_cpu_comparison.png'")
    plt.show()
    
except Exception as e:
    print(f"⚠️ Could not create plot: {e}")

print("\\n" + "="*60)
print("✅ MULTI-CPU TEST COMPLETED SUCCESSFULLY!")
print("="*60)
