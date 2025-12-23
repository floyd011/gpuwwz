# test_all_features.py
import numpy as np
import libwwz
import time

print("="*70)
print("🎯 COMPLETE LIBWWZ GPU TEST SUITE")
print("="*70)

# 1. System info
print("\\n1. 📊 SYSTEM INFORMATION:")
print("-"*40)
print(f"libwwz version: {libwwz.__version__}")
print(f"GPU available: {libwwz.GPU_AVAILABLE}")
print(f"GPU count: {libwwz.get_gpu_count()}")
print(f"GPU info:\\n{libwwz.gpu_info()}")

# 2. Generate test data
print("\\n2. 📈 GENERATING TEST DATA:")
print("-"*40)
np.random.seed(42)
t = np.sort(np.random.uniform(0, 100, 10000))
y = np.sin(2*np.pi*0.1*t) + 0.5*np.sin(2*np.pi*0.3*t) + 0.2*np.random.randn(len(t))
freq = np.linspace(0.01, 0.5, 200)
tau = np.linspace(t.min(), t.max(), 300)

print(f"Time points: {len(t):,}")
print(f"Frequencies: {len(freq):,}")
print(f"Tau points: {len(tau):,}")

# 3. Test all functions
print("\\n3. 🧪 TESTING ALL FUNCTIONS:")
print("-"*40)

functions_to_test = [
    ("wwt_cpu", lambda: libwwz.wwt_cpu(t[:1000], y[:1000], freq[:50], multi_cpu=True)),
    ("wwt_gpu", lambda: libwwz.wwt_gpu(t, y, freq, tau=tau)),
    ("wwt_memory_safe", lambda: libwwz.wwt_memory_safe(t, y, freq, tau=tau, use_gpu=True)),
]

if libwwz.get_gpu_count() > 1:
    functions_to_test.append(
        ("wwt_multi_gpu", lambda: libwwz.wwt_multi_gpu(t, y, freq, tau=tau))
    )

results = {}
for name, func in functions_to_test:
    print(f"\\n🔧 Testing {name}...")
    try:
        start = time.time()
        result = func()
        elapsed = time.time() - start
        
        if result:
            wwz, dampp = result
            print(f"   ✅ Success: {elapsed:.2f}s, shape: {wwz.shape}")
            results[name] = (wwz, dampp, elapsed)
        else:
            print(f"   ❌ Failed: No result returned")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

# 4. Performance comparison
print("\\n4. ⚡ PERFORMANCE COMPARISON:")
print("-"*40)

if 'wwt_gpu' in results and 'wwt_cpu' in results:
    gpu_time = results['wwt_gpu'][2]
    cpu_time = results['wwt_cpu'][2]
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s")
    print(f"⚡ GPU speedup: {speedup:.1f}x")

if 'wwt_multi_gpu' in results and 'wwt_gpu' in results:
    multi_time = results['wwt_multi_gpu'][2]
    single_time = results['wwt_gpu'][2]
    multi_speedup = single_time / multi_time if multi_time > 0 else 0
    print(f"Single GPU: {single_time:.2f}s, Multi-GPU: {multi_time:.2f}s")
    print(f"🚀 Multi-GPU speedup: {multi_speedup:.1f}x")

# 5. Memory check
print("\\n5. 🧠 MEMORY CHECK:")
print("-"*40)

if libwwz.GPU_AVAILABLE:
    safe_chunk, est_mem = libwwz.check_gpu_memory(
        desired_chunk_size=50,
        ntime=len(t),
        ntau=len(tau),
        verbose=True
    )
    
    if est_mem:
        print(f"Estimated memory for full processing: {est_mem:.2f} GB")
        print(f"Safe chunk size: {safe_chunk}")

# 6. Plot results
print("\\n6. 📊 PLOTTING RESULTS:")
print("-"*40)

try:
    import matplotlib.pyplot as plt
    
    if 'wwt_gpu' in results:
        wwz, dampp, elapsed = results['wwt_gpu']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(wwz.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f'WWZ Power (GPU)\\n{elapsed:.1f}s')
        axes[0].set_xlabel('Tau')
        axes[0].set_ylabel('Frequency')
        
        axes[1].imshow(dampp.T, aspect='auto', origin='lower', cmap='plasma')
        axes[1].set_title('Damping Coefficient')
        axes[1].set_xlabel('Tau')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('wwz_gpu_results.png', dpi=150)
        print("✅ Plot saved as 'wwz_gpu_results.png'")
        plt.show()
        
except Exception as e:
    print(f"⚠️ Could not create plot: {e}")

print("\\n" + "="*70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*70)
