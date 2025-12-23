import numpy as np
import libwwz
import time

print("="*60)
print("TEST SA MEMORIJSKOM OPTIMIZACIJOM")
print("="*60)

# Generiši podatke različite veličine
sizes = [
    (1000, 100, 50),   # mali
    (5000, 200, 100),  # srednji
    (10000, 300, 200)  # veliki
]

for ntime, nfreq, ntau in sizes:
    print(f"\n{'='*40}")
    print(f"Test: ntime={ntime}, nfreq={nfreq}, ntau={ntau}")
    print(f"{'='*40}")
    
    t = np.linspace(0, 100, ntime)
    y = np.sin(2*np.pi*0.1*t) + 0.3*np.random.randn(ntime)
    freq = np.linspace(0.01, 0.5, nfreq)
    tau = np.linspace(0, 100, ntau) if ntau != len(t) else t
    
    if libwwz.GPU_AVAILABLE:
        # Proveri memoriju pre izvršenja
        safe_chunk, est_mem = libwwz.check_gpu_memory(
            desired_chunk_size=50,
            ntime=ntime,
            ntau=ntau,
            verbose=True
        )
        
        if est_mem and est_mem > 6:  # Ako treba >6GB
            print(f"⚠ Upozorenje: Potrebno {est_mem:.2f} GB")
            print("Koristim automatsku optimizaciju...")
        
        # Test 1: Automatska optimizacija
        print("\nTest 1: Automatska optimizacija (optimize_memory=True)")
        start = time.time()
        try:
            wwz1, damp1 = libwwz.wwt_gpu(
                t, y, freq, tau=tau,
                gpu_chunk_size=50,  # Pokušaj 50
                optimize_memory=True  # Dozvoli automatsko podešavanje
            )
            print(f"✓ Uspešno! Vreme: {time.time()-start:.2f}s")
            print(f"  Shape: {wwz1.shape}")
        except Exception as e:
            print(f"✗ Greška: {e}")
        
        # Test 2: Manualni mali chunk
        print("\nTest 2: Manualni mali chunk (size=10)")
        start = time.time()
        try:
            wwz2, damp2 = libwwz.wwt_gpu(
                t, y, freq, tau=tau,
                gpu_chunk_size=10,  # Mali chunk
                optimize_memory=False
            )
            print(f"✓ Uspešno! Vreme: {time.time()-start:.2f}s")
        except Exception as e:
            print(f"✗ Greška: {e}")
        
        # Test 3: Procesiranje po jedna frekvencija
        print("\nTest 3: Po jedna frekvencija (chunk_size=1)")
        start = time.time()
        try:
            wwz3, damp3 = libwwz.wwt_gpu(
                t, y, freq, tau=tau,
                gpu_chunk_size=1,  # Minimalno
                optimize_memory=False
            )
            print(f"✓ Uspešno! Vreme: {time.time()-start:.2f}s")
        except Exception as e:
            print(f"✗ Greška: {e}")
            print("Probaj CPU verziju...")
            wwz_cpu, damp_cpu = libwwz.wwt_cpu(t, y, freq, tau=tau)
            print(f"CPU uspešno: {wwz_cpu.shape}")
    
    else:
        print("GPU nije dostupan, testiram CPU...")
        start = time.time()
        wwz_cpu, damp_cpu = libwwz.wwt_cpu(t, y, freq, tau=tau)
        print(f"CPU vreme: {time.time()-start:.2f}s")

print(f"\n{'='*60}")
print("PREPORUKE ZA PRODUKCIJU:")
print("1. Koristi float32 umesto float64 (dovoljna preciznost)")
print("2. Početni chunk_size=10-20 za velike dataset-ove")
print("3. Uključi optimize_memory=True za automatsko podešavanje")
print("4. Za ntime>10000, razmotri downsampling ili chunk-ovanje po vremenu")
print(f"{'='*60}")
