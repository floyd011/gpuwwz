import numpy as np
import libwwz
import time

print("=" * 60)
print(f"libwwz version: {libwwz.__version__}")
print(f"GPU available: {libwwz.GPU_AVAILABLE}")
print(f"GPU info:\n{libwwz.gpu_info()}")
print("=" * 60)

# Generiši test podatke
np.random.seed(42)
n_points = 5000
t = np.sort(np.random.uniform(0, 100, n_points))
y = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.3 * t)
y += 0.2 * np.random.randn(n_points)
freq = np.linspace(0.01, 0.5, 200)

print(f"\n📊 Test podaci:")
print(f"  Vremenskih tačaka: {len(t):,}")
print(f"  Frekvencija: {len(freq):,}")
print(f"  Tau tačaka: {len(t):,} (default)")

# Test 1: CPU
print("\n" + "=" * 60)
print("TEST 1: CPU VERZIJA")
print("=" * 60)

start = time.time()
wwz_cpu, damp_cpu = libwwz.wwt_cpu(t, y, freq)
cpu_time = time.time() - start

print(f"⏱️  CPU vreme: {cpu_time:.2f} sekundi")
print(f"📐 WWZ shape: {wwz_cpu.shape}")
print(f"📈 WWZ min/max: {wwz_cpu.min():.3f} / {wwz_cpu.max():.3f}")

# Test 2: GPU sa memorijskom optimizacijom (ako je dostupan)
if libwwz.GPU_AVAILABLE:
    print("\n" + "=" * 60)
    print("TEST 2: GPU VERZIJA SA MEMORIJSKOM OPTIMIZACIJOM")
    print("=" * 60)
    
    # Postavi GPU uređaj
    libwwz.set_gpu_device(0)
    
    # Proveri memoriju pre pokretanja
    ntime = len(t)
    ntau = len(t)  # jer je tau=None (koristi t)
    
    print("\n🔍 Provera memorije pre pokretanja:")
    print("-" * 40)
    
    # Testiraj različite željene chunk veličine
    desired_chunks = [50, 100, 200]
    
    for desired_chunk in desired_chunks:
        print(f"\n🎯 Željeni chunk size: {desired_chunk}")
        
        # Proveri da li će stati u memoriju
        safe_chunk, est_mem = libwwz.check_gpu_memory(
            desired_chunk_size=desired_chunk,
            ntime=ntime,
            ntau=ntau,
            verbose=False  # Ne prikazuj detalje ovde
        )
        
        if est_mem:
            print(f"   Procijenjena memorija: {est_mem:.2f} GB")
        
        if safe_chunk < desired_chunk:
            print(f"   ⚠️  Chunk size će biti smanjen na: {safe_chunk}")
        else:
            print(f"   ✓ Chunk size OK: {safe_chunk}")
    
    print("\n" + "=" * 60)
    print("🚀 POKRETANJE GPU TESTOVA")
    print("=" * 60)
    
    # Testiraj različite podešavanja
    test_configs = [
        {"name": "Auto-optimizacija", "chunk_size": 50, "optimize_memory": True},
        {"name": "Mali chunk", "chunk_size": 10, "optimize_memory": False},
        {"name": "Srednji chunk", "chunk_size": 20, "optimize_memory": True},
        {"name": "Po jedna freq", "chunk_size": 1, "optimize_memory": False},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 {config['name']}:")
        print(f"   Chunk size: {config['chunk_size']}")
        print(f"   Optimize memory: {config['optimize_memory']}")
        
        try:
            # Proveri memoriju pre nego što pokreneš
            safe_chunk, est_mem = libwwz.check_gpu_memory(
                desired_chunk_size=config['chunk_size'],
                ntime=ntime,
                ntau=ntau,
                verbose=False
            )
            
            # Ako optimize_memory=True, koristi safe_chunk
            actual_chunk = safe_chunk if config['optimize_memory'] else config['chunk_size']
            
            if config['optimize_memory'] and actual_chunk != config['chunk_size']:
                print(f"   ⚙️  Automatski podešeno na: {actual_chunk}")
            
            # Očisti GPU memoriju pre testa
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            
            # Pokreni test
            start = time.time()
            wwz_gpu, damp_gpu = libwwz.wwt_gpu(
                t, y, freq, 
                gpu_chunk_size=actual_chunk,
                optimize_memory=config['optimize_memory']
            )
            gpu_time = time.time() - start
            
            # Proveri tačnost
            max_diff = np.max(np.abs(wwz_cpu - wwz_gpu))
            
            # Računaj ubrzanje
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            print(f"   ✅ GPU vreme: {gpu_time:.2f} sekundi")
            print(f"   🚀 Ubrzanje: {speedup:.1f}x")
            print(f"   📏 Tačnost (max diff): {max_diff:.2e}")
            
            if max_diff < 1e-5:
                print(f"   ✓ Rezultati se poklapaju")
            else:
                print(f"   ⚠️  Razlika prevelika")
            
            results.append({
                "name": config['name'],
                "chunk": actual_chunk,
                "time": gpu_time,
                "speedup": speedup,
                "accuracy": max_diff
            })
            
        except Exception as e:
            print(f"   ❌ Greška: {str(e)[:100]}...")
            results.append({
                "name": config['name'],
                "chunk": config['chunk_size'],
                "time": None,
                "speedup": None,
                "error": str(e)
            })
        
        # Mali delay između testova
        time.sleep(0.5)
    
    # Prikaz rezultata
    print("\n" + "=" * 60)
    print("📊 REZULTATI UPOREDNO")
    print("=" * 60)
    
    print(f"\nCPU vreme: {cpu_time:.2f}s (bazna linija)")
    print("-" * 60)
    
    for result in results:
        if result.get("time"):
            print(f"{result['name']:20} | Chunk: {result['chunk']:3} | "
                  f"Vreme: {result['time']:6.2f}s | "
                  f"Ubrzanje: {result['speedup']:5.1f}x | "
                  f"Tačnost: {result['accuracy']:.1e}")
        else:
            print(f"{result['name']:20} | Chunk: {result['chunk']:3} | "
                  f"❌ {result.get('error', 'Unknown error')}")
    
    # Test 3: Original wwt funkcija sa use_gpu parametrom
    print("\n" + "=" * 60)
    print("TEST 3: ORIGINAL WWT() FUNKCIJA")
    print("=" * 60)
    
    try:
        # Očisti memoriju pre testa
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        start = time.time()
        wwz_gpu2, damp_gpu2 = libwwz.wwt(
            t, y, freq, 
            use_gpu=True, 
            gpu_chunk_size=20,
            optimize_memory=True
        )
        gpu_time2 = time.time() - start
        
        print(f"⏱️  Original wwt(use_gpu=True) vreme: {gpu_time2:.2f} sekundi")
        print(f"🚀 Ubrzanje vs CPU: {cpu_time/gpu_time2:.1f}x")
        
        # Uporedi sa wwt_gpu()
        diff_gpu_methods = np.max(np.abs(wwz_gpu - wwz_gpu2))
        print(f"📏 Razlika između wwt_gpu() i wwt(use_gpu=True): {diff_gpu_methods:.2e}")
        
    except Exception as e:
        print(f"❌ Greška: {e}")
    
    # Najbolji rezultat
    successful_results = [r for r in results if r.get("time")]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x["time"])
        print("\n" + "=" * 60)
        print("🏆 NAJBOLJI REZULTAT")
        print("=" * 60)
        print(f"Metoda: {best_result['name']}")
        print(f"Chunk size: {best_result['chunk']}")
        print(f"Vreme: {best_result['time']:.2f}s")
        print(f"Ubrzanje vs CPU: {best_result['speedup']:.1f}x")
    
else:
    print("\n" + "=" * 60)
    print("❌ GPU NIJE DOSTUPAN!")
    print("=" * 60)
    print("Instaliraj CuPy za GPU podršku:")
    print("  pip install cupy-cuda11x  # za CUDA 11.x")
    print("  pip install cupy-cuda12x  # za CUDA 12.x")
    print("\nIli koristi CPU verziju:")
    print("  result = libwwz.wwt_cpu(t, y, freq)")
    print("=" * 60)

print("\n" + "=" * 60)
print("🛠️  DOSTUPNE FUNKCIJE")
print("=" * 60)
print(f"  libwwz.wwt()              - osnovna funkcija")
print(f"  libwwz.wwt_gpu()          - GPU verzija (convenience)")
print(f"  libwwz.wwt_cpu()          - CPU verzija (convenience)")
print(f"  libwwz.GPU_AVAILABLE      - status GPU-a")
print(f"  libwwz.gpu_info()         - informacije o GPU-u")
print(f"  libwwz.set_gpu_device()   - postavi GPU uređaj")
print(f"  libwwz.check_gpu_memory() - proveri GPU memoriju")
print("\n🎯 PREPORUKE:")
print(f"  1. Uvek koristi optimize_memory=True za velike dataset-ove")
print(f"  2. Početni chunk_size=10-20")
print(f"  3. check_gpu_memory() pre pokretanja za proveru")
print("=" * 60)

# Dodatna analiza za veće dataset-ove
print("\n📈 PREDVIDANJE ZA VEĆE DATASET-OVE:")
print("-" * 40)

sizes_to_predict = [
    (10000, 500, "Srednji"),
    (20000, 1000, "Veliki"),
    (50000, 2000, "Ekstremni")
]

if libwwz.GPU_AVAILABLE:
    for n_points_pred, n_freq_pred, label in sizes_to_predict:
        safe_chunk, est_mem = libwwz.check_gpu_memory(
            desired_chunk_size=20,
            ntime=n_points_pred,
            ntau=n_points_pred,
            verbose=False
        )
        
        if est_mem:
            status = "✅ MOŽE" if est_mem < 7 else "⚠️  MARGINALNO" if est_mem < 7.5 else "❌ PREVELIKO"
            print(f"{label:10} ({n_points_pred:,} pts, {n_freq_pred} freq):")
            print(f"  {status} - Procijenjeno: {est_mem:.1f} GB, Safe chunk: {safe_chunk}")
