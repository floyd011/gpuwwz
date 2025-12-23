"""
libwwz - Weighted Wavelet Z-Transform library with GPU acceleration
"""

import numpy as np
from .wwz import (
    wwt,
    wwt_memory_safe,
    wwt_multi_gpu,
    GPU_AVAILABLE,
    wwt_multi_cpu,  # DODAJ OVO
    wwt_cpu_worker, # DODAJ OVO
    get_gpu_count,
    get_gpu_info_all,
    get_cpu_count,   # DODAJ OVO
    set_gpu_device,
    check_gpu_memory
)

# ====================== CONVENIENCE FUNCTIONS ======================
def wwt_cpu(t, y, freq, tau=None, c=1/(8*np.pi**2), 
            decay_constant=0.01, return_coefs=True,
            multi_cpu=False, n_workers=None):
    """
    WWT bez GPU (samo CPU) sa multi-core podrškom.
    
    Args:
        multi_cpu: Koristi sve CPU jezgre
        n_workers: Broj CPU workers (None = auto)
    """
    if multi_cpu:
        return wwt_multi_cpu(t, y, freq, tau=tau, c=c, n_workers=n_workers)
    else:
        return wwt(t, y, freq, tau=tau, c=c,
                   decay_constant=decay_constant,
                   return_coefs=return_coefs,
                   use_gpu=False)

def system_info():
    """Vrati informacije o sistemu."""
    info = []
    info.append(f"🎯 libwwz version: {__version__}")
    info.append(f"🎮 GPU available: {GPU_AVAILABLE}")
    
    if GPU_AVAILABLE:
        info.append(f"📊 GPU count: {get_gpu_count()}")
    
    info.append(f"🖥️ CPU cores: {get_cpu_count()}")
    info.append(f"📈 NumPy version: {np.__version__}")
    
    return "\\n".join(info)

def gpu_info():
    """
    Vrati informacije o svim GPU uređajima.
    """
    try:
        info = get_gpu_info_all()
        if info:
            header = f"🎮 Available GPUs ({get_gpu_count()} devices):"
            return f"{header}\\n" + "\\n".join([f"  {line}" for line in info])
        else:
            return "No GPU devices found"
    except Exception as e:
        return f"Error getting GPU info: {e}"

def wwt_gpu(t, y, freq, tau=None, c=1/(8*np.pi**2), 
            decay_constant=0.01, return_coefs=True, 
            gpu_chunk_size=None, optimize_memory=True,
            multi_gpu=False):
    """
    WWT sa GPU ubrzanjem (convenience funkcija).
    
    Args:
        multi_gpu: Koristi sve dostupne GPU-ove ako je True
    """
    if multi_gpu and GPU_AVAILABLE and get_gpu_count() > 1:
        return wwt_multi_gpu(t, y, freq, tau=tau, c=c, use_gpus=True)
    else:
        return wwt(t, y, freq, tau=tau, c=c,
                   decay_constant=decay_constant,
                   return_coefs=return_coefs,
                   use_gpu=True,
                   gpu_chunk_size=gpu_chunk_size,
                   optimize_memory=optimize_memory)

def safe_convert_to_numpy(array):
    """
    Bezbedno konvertuj bilo koji array u NumPy.
    """
    # Ako je već NumPy
    if isinstance(array, np.ndarray):
        return array
    
    # Ako je CuPy
    try:
        if hasattr(array, 'get'):
            return array.get()
    except:
        pass
    
    # Ako je nešto drugo
    try:
        return np.asarray(array)
    except:
        raise ValueError(f"Cannot convert {type(array)} to NumPy array")

# ====================== VERSION AND INFO ======================

__version__ = "1.5.0"
__author__ = "libwwz team with GPU enhancements"
__email__ = ""

__all__ = [
    'wwt',
    'wwt_gpu',
    'wwt_cpu',
    'wwt_memory_safe',
    'wwt_multi_gpu',
    'GPU_AVAILABLE',
    'gpu_info',
    'get_gpu_count',
    'get_gpu_info_all',
    'set_gpu_device',
    'check_gpu_memory',
    'safe_convert_to_numpy'
]
__all__.extend([
    'wwt_multi_cpu',
    'wwt_cpu_worker',
    'get_cpu_count',
    'system_info'
])
try:
    from .plot_methods import (
        linear_plotter,
        octave_plotter,
        plot_wwz_comprehensive,
        plot_modern_wwz,
        prepare_wwz_data
    )
    
    __all__.extend([
        'linear_plotter',
        'octave_plotter', 
        'plot_wwz_comprehensive',
        'plot_modern_wwz',
        'prepare_wwz_data'
    ])
except ImportError:
    # plot_methods možda ne postoji
    pass

# ====================== STARTUP MESSAGE ======================

if __name__ != "__main__":
    print(f"🚀 libwwz {__version__} loaded successfully")
    print(f"🎯 GPU acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")
  
    cpu_count = get_cpu_count()
    print(f"🖥️ CPU cores: {cpu_count}")
    print(f"   Multi-CPU support: ✅ ENABLED (up to {cpu_count} cores)")
      
    if GPU_AVAILABLE:
        gpu_count = get_gpu_count()
        if gpu_count > 0:
            print(f"🎮 Available GPUs: {gpu_count}")
            if gpu_count > 1:
                print("   Multi-GPU support: ✅ ENABLED")
            else:
                print("   Multi-GPU support: ⚠️  Only 1 GPU available")
    
    print("🔧 Available functions:")
    print("   • wwt(), wwt_gpu(), wwt_cpu() - basic WWZ")
    print("   • wwt_memory_safe() - memory optimized")
    print("   • wwt_multi_gpu() - multi-GPU support")
    print("   • gpu_info(), set_gpu_device() - GPU utilities")
    print("   • wwt_multi_cpu() - multi-CPU support")
    print("=" * 60)
