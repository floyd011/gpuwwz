import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
# CUDA/GPU detekcija
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy loaded successfully. GPU acceleration available.")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("⚠️ CuPy not installed. Using CPU-only mode.")

# ====================== GPU UTILITY FUNKCIJE ======================
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import concurrent.futures

def get_cpu_count():
    """Vrati broj dostupnih CPU jezgri."""
    try:
        return cpu_count()
    except:
        return 1

def wwt_multi_cpu(t, y, freq, tau=None, c=1/(8*np.pi**2), 
                  n_workers=None, chunk_size=10, **kwargs):
    """
    WWZ sa multi-core CPU podrškom koristeći multiprocessing.
    
    Args:
        n_workers: Broj CPU jezgri za korišćenje (None = sve dostupne)
        chunk_size: Koliko frekvencija da se procesira u jednom batch-u
    """
    if n_workers is None:
        n_workers = max(1, get_cpu_count() - 1)  # Ostavi jednu jezgru slobodnu
    
    print(f"🖥️ Multi-CPU mode: Using {n_workers} cores")
    
    # Podeli frekvencije na chunk-ove za procesiranje
    nfreq = len(freq)
    freq_chunks = []
    
    for i in range(0, nfreq, chunk_size):
        end_idx = min(i + chunk_size, nfreq)
        freq_chunks.append(freq[i:end_idx])
    
    print(f"📊 Split {nfreq} frequencies into {len(freq_chunks)} chunks")
    
    # Pripremi shared arguments
    if tau is None:
        tau_array = t
    else:
        tau_array = tau
    
    # Funkcija za procesiranje chunk-a
    def process_chunk(chunk_freq):
        """Procesira jedan chunk frekvencija."""
        try:
            # Koristi numpy direktno za CPU processing
            wwz_chunk, dampp_chunk = wwt_cpu_worker(
                t, y, chunk_freq, tau_array, c
            )
            return wwz_chunk, dampp_chunk
        except Exception as e:
            print(f"⚠️ Chunk processing error: {e}")
            return None
    
    # Paralelno procesiranje sa ThreadPoolExecutor (bolje za numpy)
    print("🚀 Starting parallel processing...")
    start_time = time.time()
    
    try:
        # Koristi ThreadPoolExecutor za I/O bound operacije
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit sve chunk-ove
            future_to_chunk = {
                executor.submit(process_chunk, chunk): idx 
                for idx, chunk in enumerate(freq_chunks)
            }
            
            # Prikupi rezultate
            results = []
            completed = 0
            total = len(freq_chunks)
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append((chunk_idx, result))
                    
                    completed += 1
                    if completed % max(1, total // 10) == 0:
                        print(f"  Progress: {completed}/{total} chunks")
                        
                except Exception as e:
                    print(f"❌ Error in chunk {chunk_idx}: {e}")
        
        # Sortiraj rezultate po indeksu
        results.sort(key=lambda x: x[0])
        
        # Kombinuj sve chunk-ove
        all_wwz = []
        all_dampp = []
        
        for _, (wwz_chunk, dampp_chunk) in results:
            all_wwz.append(wwz_chunk)
            all_dampp.append(dampp_chunk)
        
        if all_wwz:
            wwz_combined = np.concatenate(all_wwz, axis=0)
            dampp_combined = np.concatenate(all_dampp, axis=0)
            
            elapsed = time.time() - start_time
            print(f"✅ Multi-CPU completed in {elapsed:.2f}s")
            print(f"📏 Combined shape: {wwz_combined.shape}")
            
            return wwz_combined, dampp_combined
        else:
            raise Exception("No chunks processed successfully")
            
    except Exception as e:
        print(f"❌ Multi-CPU failed: {e}")
        print("🔄 Falling back to single CPU...")
        return wwt_memory_safe(t, y, freq, tau=tau, c=c, use_gpu=False, **kwargs)

def wwt_cpu_worker(t, y, freq_chunk, tau, c):
    """
    Worker funkcija za CPU processing (koristi se u multiprocessing).
    """
    import numpy as np
    
    nfreq = len(freq_chunk)
    ntau = len(tau)
    
    wwz_chunk = np.empty((nfreq, ntau), dtype=np.float32)
    dampp_chunk = np.empty((nfreq, ntau), dtype=np.float32)
    
    for i, f in enumerate(freq_chunk):
        omega = 2 * np.pi * f
        decay = omega / (2 * np.pi * c)
        
        # Vektorizovano po tau
        t_expanded = t[:, np.newaxis]
        tau_expanded = tau[np.newaxis, :]
        
        t_tau = t_expanded - tau_expanded
        weight = np.exp(-decay * (t_tau ** 2))
        wavelet = np.exp(1j * omega * t_tau)
        
        y_expanded = y[:, np.newaxis]
        y_weighted = y_expanded * weight
        wavelet_weighted = wavelet * weight
        
        # Calculations
        sum_wavelet_y = np.sum(wavelet_weighted * np.conj(y_weighted), axis=0)
        num = np.abs(sum_wavelet_y) ** 2
        
        sum_weight = np.sum(weight, axis=0)
        sum_weight_y2 = np.sum(weight * np.abs(y_expanded) ** 2, axis=0)
        denom = sum_weight * sum_weight_y2
        
        # WWZ values
        wwz_row = np.zeros(ntau, dtype=np.float32)
        mask = denom != 0
        wwz_row[mask] = num[mask] / denom[mask]
        
        # Damping
        dampp_row = np.sqrt(sum_weight / 2)
        
        # Store
        wwz_chunk[i, :] = wwz_row
        dampp_chunk[i, :] = dampp_row
    
    return wwz_chunk, dampp_chunk

def get_gpu_count():
    """Vrati broj dostupnih GPU uređaja."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except:
        return 0

def get_gpu_info_all():
    """Vrati informacije o svim GPU uređajima."""
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        info = []
        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props.get('name', b'Unknown').decode()
            memory_gb = props.get('totalGlobalMem', 0) / 1e9
            compute = props.get('major', 0), props.get('minor', 0)
            info.append(f"GPU {i}: {name} ({memory_gb:.2f} GB, CUDA {compute[0]}.{compute[1]})")
        return info
    except:
        return ["GPU info not available"]

def set_gpu_device(device_id=0):
    """Postavi aktivni GPU uređaj."""
    try:
        import cupy as cp
        cp.cuda.Device(device_id).use()
        print(f"✅ GPU device set to: {device_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to set GPU device {device_id}: {e}")
        return False

# ====================== MEMORY CHECK FUNKCIJE ======================

def check_gpu_memory(desired_chunk_size, ntime, ntau, verbose=True):
    """
    Check if chunk can fit in GPU memory.
    
    Returns:
        tuple: (safe_chunk_size, estimated_memory_gb)
    """
    if not GPU_AVAILABLE:
        if verbose:
            print("⚠️ GPU not available")
        return min(10, desired_chunk_size), None
    
    try:
        import cupy as cp
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        free_gb = free_mem / 1e9
        total_gb = total_mem / 1e9
        
        # Proračun memorije za jednu frekvenciju
        # t_tau, weight, wavelet - svaki: ntime × ntau × 4 bytes
        mem_per_freq = ntime * ntau * 4 * 3  # 3 float32 arrays
        mem_per_freq_gb = mem_per_freq / 1e9
        
        # Maksimalni safe chunk size (koristi 70% slobodne memorije)
        max_safe_chunk = int((free_gb * 0.7) / mem_per_freq_gb)
        safe_chunk = min(desired_chunk_size, max_safe_chunk, 50)  # Max 50
        
        if verbose:
            print(f"🎮 GPU Memory: {free_gb:.2f} GB free of {total_gb:.2f} GB")
            print(f"📊 Memory per frequency: {mem_per_freq_gb:.3f} GB")
            print(f"🎯 Maximum safe chunk size: {max_safe_chunk}")
            print(f"⚙️ Recommended chunk size: {safe_chunk}")
        
        return max(1, safe_chunk), mem_per_freq_gb * desired_chunk_size
        
    except Exception as e:
        print(f"⚠️ Cannot check GPU memory: {e}")
        return min(10, desired_chunk_size), None

# ====================== WWZ FUNKCIJE ======================

def wwt(t, y, freq, tau=None, c=1/(8*np.pi**2), decay_constant=0.01,
        return_coefs=True, use_gpu=False, gpu_chunk_size=None, 
        optimize_memory=True):
    """
    Original WWZ funkcija sa GPU podrškom.
    """
    # Odaberi backend
    if use_gpu and GPU_AVAILABLE:
        xp = cp
        t = cp.asarray(t, dtype=cp.float32)
        y = cp.asarray(y, dtype=cp.float32)
        freq = cp.asarray(freq, dtype=cp.float32)
    else:
        xp = np
        t = np.asarray(t, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        freq = np.asarray(freq, dtype=np.float64)
    
    if tau is None:
        tau = t
    
    if use_gpu and GPU_AVAILABLE:
        tau = cp.asarray(tau, dtype=cp.float32)
    else:
        tau = np.asarray(tau, dtype=np.float64)
    
    nfreq = len(freq)
    ntau = len(tau)
    ntime = len(t)
    
    wwz = xp.empty((nfreq, ntau), dtype=xp.float32)
    dampp = xp.empty((nfreq, ntau), dtype=xp.float32)
    
    # GPU IMPLEMENTACIJA
    if use_gpu and GPU_AVAILABLE:
        # Automatsko određivanje chunk size
        if gpu_chunk_size is None:
            gpu_chunk_size = min(20, nfreq)
        
        # Optimizacija memorije
        if optimize_memory:
            try:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                free_mem_gb = free_mem / 1e9
                
                mem_per_freq = ntime * ntau * 4 * 3
                mem_per_freq_gb = mem_per_freq / 1e9
                
                max_safe_chunk = int((free_mem_gb * 0.7) / mem_per_freq_gb)
                max_safe_chunk = max(1, min(max_safe_chunk, nfreq))
                
                if max_safe_chunk < gpu_chunk_size:
                    print(f"🎯 GPU memory optimization: Reducing chunk size from {gpu_chunk_size} to {max_safe_chunk}")
                    gpu_chunk_size = max_safe_chunk
                
            except Exception as e:
                print(f"⚠️ Memory check failed: {e}. Using default chunk size {gpu_chunk_size}")
        
        print(f"⚙️ Processing {nfreq} frequencies in chunks of {gpu_chunk_size}")
        
        # Procesiraj po chunk-ovima
        for start_idx in range(0, nfreq, gpu_chunk_size):
            end_idx = min(start_idx + gpu_chunk_size, nfreq)
            chunk_freq = freq[start_idx:end_idx]
            
            # Procesiraj svaku frekvenciju u chunk-u
            for idx_in_chunk, f in enumerate(chunk_freq):
                i = start_idx + idx_in_chunk
                omega = 2 * xp.pi * f
                decay = omega / (2 * xp.pi * c)
                
                # VEKTORIZOVANO PO TAU
                t_expanded = t[:, xp.newaxis]
                tau_expanded = tau[xp.newaxis, :]
                
                t_tau = t_expanded - tau_expanded
                weight = xp.exp(-decay * (t_tau ** 2))
                wavelet = xp.exp(1j * omega * t_tau)
                
                y_expanded = y[:, xp.newaxis]
                y_weighted = y_expanded * weight
                wavelet_weighted = wavelet * weight
                
                # Sums
                sum_wavelet_y = xp.sum(wavelet_weighted * xp.conj(y_weighted), axis=0)
                num = xp.abs(sum_wavelet_y) ** 2
                
                sum_weight = xp.sum(weight, axis=0)
                sum_weight_y2 = xp.sum(weight * xp.abs(y_expanded) ** 2, axis=0)
                denom = sum_weight * sum_weight_y2
                
                # WWZ values
                wwz_row = xp.zeros(ntau, dtype=xp.float32)
                mask = denom != 0
                wwz_row[mask] = num[mask] / denom[mask]
                
                # Damping
                dampp_row = xp.sqrt(sum_weight / 2)
                
                # Store
                wwz[i, :] = wwz_row
                dampp[i, :] = dampp_row
            
            # Očisti memoriju između chunk-ova
            if start_idx + gpu_chunk_size < nfreq:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    
    # CPU IMPLEMENTACIJA
    else:
        print(f"🖥️ CPU processing: {nfreq} frequencies × {ntau} tau points")
        
        for i in range(nfreq):
            f = freq[i]
            omega = 2 * xp.pi * f
            decay = omega / (2 * xp.pi * c)
            
            for j in range(ntau):
                t_j = tau[j]
                t_tau = t - t_j
                weight = xp.exp(-decay * (t_tau ** 2))
                wavelet = xp.exp(1j * omega * t_tau)
                
                y_weighted = y * weight
                wavelet_weighted = wavelet * weight
                
                num = xp.abs(xp.sum(wavelet_weighted * xp.conj(y_weighted))) ** 2
                denom = xp.sum(weight) * xp.sum(weight * xp.abs(y) ** 2)
                
                wwz[i, j] = (num / denom) if denom != 0 else 0
                dampp[i, j] = xp.sqrt(xp.sum(weight) / 2)
            
            # Progress
            if (i + 1) % max(1, nfreq // 10) == 0:
                print(f"  Progress: {i+1}/{nfreq} frequencies")
    
    if return_coefs:
        if use_gpu and GPU_AVAILABLE:
            # Uvek vraćaj NumPy
            wwz_cpu = cp.asnumpy(wwz) if hasattr(wwz, 'get') else wwz
            dampp_cpu = cp.asnumpy(dampp) if hasattr(dampp, 'get') else dampp
            cp.get_default_memory_pool().free_all_blocks()
            return wwz_cpu, dampp_cpu
        else:
            return wwz, dampp
    
    return None

def wwt_memory_safe(t, y, freq, tau=None, c=1/(8*np.pi**2), 
                    use_gpu=False, max_memory_gb=6.0, 
                    tau_downsample=True, time_downsample=False,
                    multi_gpu=False, gpu_device=0,
                    multi_cpu=False, n_cpu_workers=None):
    """
    WWT sa multi-GPU I multi-CPU podrškom.
    
    Args:
        multi_cpu: Koristi sve CPU jezgre ako je True
        n_cpu_workers: Broj CPU workers (None = auto)
    """
    # Multi-CPU mode
    if not use_gpu and multi_cpu and len(freq) > 50:
        try:
            print(f"🖥️ Multi-CPU mode activated")
            return wwt_multi_cpu(t, y, freq, tau=tau, c=c, 
                                n_workers=n_cpu_workers, chunk_size=20)
        except Exception as e:
            print(f"⚠️ Multi-CPU failed: {e}, falling back to single CPU")
    
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        
        # Postavi GPU device
        if not multi_gpu:
            cp.cuda.Device(gpu_device).use()
        
        xp = cp
        t = cp.asarray(t, dtype=cp.float32)
        y = cp.asarray(y, dtype=cp.float32)
        freq = cp.asarray(freq, dtype=cp.float32)
    else:
        xp = np
        t = np.asarray(t, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        freq = np.asarray(freq, dtype=np.float64)
    
    # Postavi tau sa optimizacijom
    if tau is None:
        tau = t
    
    # Tau downsample za memory optimization
    if tau_downsample and len(tau) > 1000:
        ntau_target = min(500, len(tau) // 10)
        tau = np.linspace(tau.min(), tau.max(), ntau_target)
        print(f"🎯 Tau downsample: {len(t)} → {len(tau)} points")
    
    if use_gpu and GPU_AVAILABLE:
        tau = cp.asarray(tau, dtype=cp.float32)
    else:
        tau = np.asarray(tau, dtype=np.float64)
    
    nfreq = len(freq)
    ntau = len(tau)
    ntime = len(t)
    
    # Multi-GPU logika
    if use_gpu and GPU_AVAILABLE and multi_gpu and nfreq > 100:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count > 1:
                print(f"🚀 Multi-GPU mode: {gpu_count} GPUs detected")
                return wwt_multi_gpu_parallel(t, y, freq, tau, c, gpu_count)
        except Exception as e:
            print(f"⚠️ Multi-GPU failed: {e}, falling back to single GPU")
    
    # Single GPU ili CPU
    wwz = xp.empty((nfreq, ntau), dtype=xp.float32)
    dampp = xp.empty((nfreq, ntau), dtype=xp.float32)
    
    print(f"⚙️ Processing: {nfreq} freq × {ntau} tau × {ntime} time points")
    
    # Procesiraj svaku frekvenciju
    for i in range(nfreq):
        f = freq[i]
        omega = 2 * xp.pi * f
        decay = omega / (2 * xp.pi * c)
        
        # Vektorizovano po tau
        t_expanded = t[:, xp.newaxis]
        tau_expanded = tau[xp.newaxis, :]
        
        t_tau = t_expanded - tau_expanded
        weight = xp.exp(-decay * (t_tau ** 2))
        wavelet = xp.exp(1j * omega * t_tau)
        
        y_expanded = y[:, xp.newaxis]
        y_weighted = y_expanded * weight
        wavelet_weighted = wavelet * weight
        
        # Calculations
        sum_wavelet_y = xp.sum(wavelet_weighted * xp.conj(y_weighted), axis=0)
        num = xp.abs(sum_wavelet_y) ** 2
        
        sum_weight = xp.sum(weight, axis=0)
        sum_weight_y2 = xp.sum(weight * xp.abs(y_expanded) ** 2, axis=0)
        denom = sum_weight * sum_weight_y2
        
        # WWZ values
        wwz_row = xp.zeros(ntau, dtype=xp.float32)
        mask = denom != 0
        wwz_row[mask] = num[mask] / denom[mask]
        
        # Damping
        dampp_row = xp.sqrt(sum_weight / 2)
        
        # Store
        wwz[i, :] = wwz_row
        dampp[i, :] = dampp_row
        
        # Progress
        if (i + 1) % max(1, nfreq // 10) == 0:
            print(f"  Progress: {i+1}/{nfreq} frequencies")
    
    # Uvek vraćaj NumPy
    if use_gpu and GPU_AVAILABLE:
        wwz = cp.asnumpy(wwz) if hasattr(wwz, 'get') else wwz
        dampp = cp.asnumpy(dampp) if hasattr(dampp, 'get') else dampp
        if not multi_gpu:
            cp.get_default_memory_pool().free_all_blocks()
    
    return wwz, dampp

def wwt_multi_gpu_parallel(t, y, freq, tau, c, gpu_count):
    """
    Paralelna multi-GPU implementacija.
    """
    import cupy as cp
    from multiprocessing import Pool
    
    nfreq = len(freq)
    
    # Podeli frekvencije na GPU-ove
    freqs_per_gpu = nfreq // gpu_count
    chunks = []
    
    for i in range(gpu_count):
        start = i * freqs_per_gpu
        if i == gpu_count - 1:
            end = nfreq
        else:
            end = (i + 1) * freqs_per_gpu
        chunks.append((i, freq[start:end]))
    
    # Funkcija za procesiranje na specifičnom GPU-u
    def process_chunk(args):
        gpu_id, chunk_freq = args
        
        try:
            # Postavi GPU
            cp.cuda.Device(gpu_id).use()
            
            # Pokreni WWZ za ovaj chunk
            result = wwt_memory_safe(
                cp.asnumpy(t) if hasattr(t, 'get') else t,
                cp.asnumpy(y) if hasattr(y, 'get') else y,
                cp.asnumpy(chunk_freq) if hasattr(chunk_freq, 'get') else chunk_freq,
                tau=cp.asnumpy(tau) if hasattr(tau, 'get') else tau,
                c=c,
                use_gpu=True,
                max_memory_gb=6.0,
                multi_gpu=False,
                gpu_device=gpu_id
            )
            
            return result
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} error: {e}")
            return None
    
    # Paralelno procesiranje
    print(f"🚀 Parallel processing on {gpu_count} GPUs")
    
    try:
        with Pool(processes=gpu_count) as pool:
            results = pool.map(process_chunk, chunks)
        
        # Kombinuj rezultate
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            wwz_parts = [r[0] for r in valid_results]
            dampp_parts = [r[1] for r in valid_results]
            
            wwz_combined = np.concatenate(wwz_parts, axis=0)
            dampp_combined = np.concatenate(dampp_parts, axis=0)
            
            return wwz_combined, dampp_combined
        else:
            raise Exception("All GPUs failed")
            
    except Exception as e:
        print(f"❌ Multi-GPU parallel failed: {e}")
        # Fallback na single GPU
        cp.cuda.Device(0).use()
        return wwt_memory_safe(t, y, freq, tau=tau, c=c, use_gpu=True, multi_gpu=False)

def wwt_multi_gpu(t, y, freq, tau=None, c=1/(8*np.pi**2), use_gpus=True, **kwargs):
    """
    WWZ sa multi-GPU podrškom (high-level API).
    """
    if not use_gpus or not GPU_AVAILABLE:
        return wwt_memory_safe(t, y, freq, tau=tau, c=c, use_gpu=True, **kwargs)
    
    try:
        import cupy as cp
        gpu_count = get_gpu_count()
        
        if gpu_count <= 1:
            print("ℹ️ Only 1 GPU available, using single GPU mode")
            return wwt_memory_safe(t, y, freq, tau=tau, c=c, use_gpu=True, **kwargs)
        
        print(f"🎮 Multi-GPU mode: {gpu_count} GPUs available")
        
        # Podeli frekvencije
        nfreq = len(freq)
        freqs_per_gpu = nfreq // gpu_count
        
        all_wwz = []
        all_dampp = []
        
        for gpu_id in range(gpu_count):
            # Postavi GPU
            cp.cuda.Device(gpu_id).use()
            
            # Odredi frekvencije
            start_idx = gpu_id * freqs_per_gpu
            if gpu_id == gpu_count - 1:
                end_idx = nfreq
            else:
                end_idx = (gpu_id + 1) * freqs_per_gpu
            
            gpu_freq = freq[start_idx:end_idx]
            
            if len(gpu_freq) == 0:
                continue
            
            print(f"   GPU {gpu_id}: processing {len(gpu_freq)} frequencies")
            
            # Pokreni WWZ
            wwz_gpu, dampp_gpu = wwt_memory_safe(
                t, y, gpu_freq, tau=tau, c=c,
                use_gpu=True,
                max_memory_gb=6.0,
                multi_gpu=False,
                gpu_device=gpu_id
            )
            
            all_wwz.append(wwz_gpu)
            all_dampp.append(dampp_gpu)
        
        # Kombinuj
        wwz_combined = np.concatenate(all_wwz, axis=0)
        dampp_combined = np.concatenate(all_dampp, axis=0)
        
        print(f"✅ Multi-GPU completed: {wwz_combined.shape}")
        
        return wwz_combined, dampp_combined
        
    except Exception as e:
        print(f"❌ Multi-GPU error: {e}, falling back to single GPU")
        return wwt_memory_safe(t, y, freq, tau=tau, c=c, use_gpu=True, **kwargs)

# ====================== EXPORT ======================

__all__ = [
    'wwt',
    'wwt_multi_cpu',
    'wwt_cpu_worker',
    'get_cpu_count',
    'wwt_memory_safe',
    'wwt_multi_gpu',
    'GPU_AVAILABLE',
    'get_gpu_count',
    'get_gpu_info_all',
    'set_gpu_device',
    'check_gpu_memory'
]
