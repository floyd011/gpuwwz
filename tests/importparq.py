import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from pathlib import Path
import time

# DODAJ OVAJ IMPORT - proveri da li je libwwz instaliran
try:
    import libwwz
    from libwwz import wwt_memory_safe, wwt_gpu, wwt_cpu, GPU_AVAILABLE
    LIBWWZ_AVAILABLE = True
except ImportError:
    print("⚠️  libwwz nije instaliran. WWZ funkcije neće biti dostupne.")
    print("   Instaliraj sa: pip install libwwz")
    LIBWWZ_AVAILABLE = False

def explore_parquet_file(file_path):
    """
    Kompletna analiza Parquet fajla.
    
    Args:
        file_path: Putanja do Parquet fajla
    """
    print("=" * 80)
    print(f"ANALIZA PARQUET FAJLA: {file_path}")
    print("=" * 80)
    
    # Provera da li fajl postoji
    if not os.path.exists(file_path):
        print(f"❌ Fajl ne postoji: {file_path}")
        return None
    
    # Provera veličine fajla
    file_size = os.path.getsize(file_path)
    print(f"📁 Veličina fajla: {file_size:,} bajtova ({file_size/1024/1024:.2f} MB)")
    
    try:
        # 1. Učitaj fajl pomoću pandas
        print("\n1. 📊 UČITAVANJE SA PANDAS:")
        print("-" * 40)
        
        df = pd.read_parquet(file_path)
        
        # Osnovne informacije
        print(f"   Broj redova: {df.shape[0]:,}")
        print(f"   Broj kolona: {df.shape[1]}")
        print(f"   Ukupno ćelija: {df.size:,}")
        
        # 2. Prikaz kolona i tipova podataka
        print("\n2. 📋 STRUKTURA KOLONA:")
        print("-" * 40)
        
        print("   ┌─────────────────────────────────────────────────────────────┐")
        print("   │   Kolona                   Tip        Nedostajućih   Jedinstvenih │")
        print("   ├─────────────────────────────────────────────────────────────┤")
        
        for i, col in enumerate(df.columns):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            # Skraćivanje imena kolona ako su predugačka
            col_display = col[:25] + "..." if len(col) > 25 else col.ljust(28)
            dtype_display = dtype[:15].ljust(15)
            
            print(f"   │ {i+1:2}. {col_display} {dtype_display} {null_count:8,} {unique_count:12,} │")
        
        print("   └─────────────────────────────────────────────────────────────┘")
        
        # 3. Statistički pregled numeričkih kolona
        print("\n3. 📈 STATISTIKE NUMERIČKIH KOLONA:")
        print("-" * 40)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                print(f"\n   🔹 {col}:")
                stats = df[col].describe()
                print(f"      Count: {stats['count']:,}")
                print(f"      Mean:  {stats['mean']:.6f}")
                print(f"      Std:   {stats['std']:.6f}")
                print(f"      Min:   {stats['min']:.6f}")
                print(f"      25%:   {stats['25%']:.6f}")
                print(f"      50%:   {stats['50%']:.6f}")
                print(f"      75%:   {stats['75%']:.6f}")
                print(f"      Max:   {stats['max']:.6f}")
        else:
            print("   ℹ️  Nema numeričkih kolona")
        
        # 4. Pregled kategorijskih/tekstualnih kolona
        print("\n4. 📝 PREGLED NE-NUMERIČKIH KOLONA:")
        print("-" * 40)
        
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                print(f"\n   🔸 {col} (tip: {df[col].dtype}):")
                
                # Prikaz prvih nekoliko jedinstvenih vrednosti
                unique_vals = df[col].dropna().unique()
                print(f"      Jedinstvenih vrednosti: {len(unique_vals):,}")
                
                if len(unique_vals) <= 10:
                    print(f"      Vrednosti: {list(unique_vals[:10])}")
                else:
                    print(f"      Prvih 10 vrednosti: {list(unique_vals[:10])}")
                    print(f"      Poslednjih 10 vrednosti: {list(unique_vals[-10:])}")
                
                # Frekvencija vrednosti
                if len(unique_vals) <= 20:
                    value_counts = df[col].value_counts().head(20)
                    print(f"      Frekvencije (top 20):")
                    for val, count in value_counts.items():
                        print(f"        '{val}': {count:,}")
        else:
            print("   ℹ️  Nema ne-numeričkih kolona")
        
        # 5. Metapodaci Parquet fajla
        print("\n5. 🔍 METAPODACI PARQUET FAJLA:")
        print("-" * 40)
        
        try:
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            
            print(f"   Broj redova u metadata: {metadata.num_rows:,}")
            print(f"   Broj row groups: {metadata.num_row_groups}")
            print(f"   Format verzija: {metadata.format_version}")
            print(f"   Kreirao: {metadata.created_by}")
            
            # Informacije o row groups
            print(f"\n   Row Groups informacije:")
            for i in range(metadata.num_row_groups):
                rg_meta = metadata.row_group(i)
                print(f"   ┌─ Row Group {i}:")
                print(f"   │  Broj redova: {rg_meta.num_rows:,}")
                print(f"   │  Ukupna veličina: {rg_meta.total_byte_size:,} bajtova")
                print(f"   │  Kolone: {rg_meta.num_columns}")
                
        except Exception as e:
            print(f"   ℹ️  Nije moguće pročitati Parquet metadata: {e}")
        
        # 6. Pronalaženje potencijalnih vremenskih i vrednosnih kolona za WWZ
        print("\n6. 🎯 PRETRAGA ZA WWZ PODATKE:")
        print("-" * 40)
        
        time_candidates = []
        value_candidates = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Potencijalne vremenske kolone
            time_keywords = ['time', 'timestamp', 'date', 't_', '_t', 'sec', 'second', 'minute', 
                            'hour', 'day', 'epoch', 'jd', 'mjd', 'utc', 'datetime']
            
            # Potencijalne vrednosne kolone
            value_keywords = ['value', 'val', 'mag', 'magnitude', 'flux', 'count', 'rate', 
                             'amplitude', 'signal', 'y', 'data', 'measurement', 'obs']
            
            if any(keyword in col_lower for keyword in time_keywords):
                time_candidates.append(col)
            
            if any(keyword in col_lower for keyword in value_keywords):
                value_candidates.append(col)
        
        print(f"   Potencijalne VREMENSKE kolone: {time_candidates}")
        print(f"   Potencijalne VREDNOSTI kolone: {value_candidates}")
        
        # 7. Prikaz prvih i poslednjih redova
        print("\n7. 👁️  PREGLED PODATAKA (head i tail):")
        print("-" * 40)
        
        print("\n   Prvih 5 redova:")
        print(df.head().to_string())
        
        print("\n   Poslednjih 5 redova:")
        print(df.tail().to_string())
        
        # 8. Kreiranje summary fajl
        print("\n8. 💾 KREIRANJE SUMMARY FAJLA:")
        print("-" * 40)
        
        summary_file = file_path.replace('.parquet', '_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"PARQUET FILE ANALYSIS: {file_path}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)\n")
            f.write(f"Rows: {df.shape[0]:,}\n")
            f.write(f"Columns: {df.shape[1]}\n\n")
            
            f.write("COLUMN STRUCTURE:\n")
            f.write("-" * 40 + "\n")
            for col in df.columns:
                f.write(f"{col}: {df[col].dtype} | Nulls: {df[col].isnull().sum():,} | Unique: {df[col].nunique():,}\n")
            
            f.write("\nFIRST 10 ROWS:\n")
            f.write("-" * 40 + "\n")
            f.write(df.head(10).to_string())
            
            f.write("\n\nLAST 10 ROWS:\n")
            f.write("-" * 40 + "\n")
            f.write(df.tail(10).to_string())
        
        print(f"   ✅ Summary sačuvan u: {summary_file}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ GREŠKA PRI ČITANJU FAJLA:")
        print(f"   {type(e).__name__}: {e}")
        
        # Dodatna dijagnostika
        print("\n🔧 DIJAGNOSTIKA:")
        print(f"   Da li je Parquet fajl: {file_path.endswith('.parquet')}")
        print(f"   Da li je file empty: {file_size == 0}")
        
        # Pokušaj sa PyArrow direktno
        try:
            print("\n🔄 Pokušavam sa PyArrow direktno...")
            table = pq.read_table(file_path)
            print(f"   ✅ Uspesno učitano sa PyArrow")
            print(f"   Broj redova: {table.num_rows:,}")
            print(f"   Broj kolona: {table.num_columns}")
            return table.to_pandas()
        except Exception as e2:
            print(f"   ❌ PyArrow takođe ne može da učita: {e2}")
        
        return None


def suggest_wwz_columns(df):
    """
    Predlaže koje kolone koristiti za WWZ analizu.
    """
    if df is None:
        return
    
    print("\n" + "=" * 80)
    print("🤖 PREPORUKE ZA WWZ ANALIZU")
    print("=" * 80)
    
    # Numeričke kolone
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("❌ Nema dovoljno numeričkih kolona za WWZ analizu (potrebne bar 2)")
        return
    
    print(f"📈 Numeričke kolone: {numeric_cols}")
    
    # Pokušaj da identifikuješ vremensku kolonu
    time_col_candidates = []
    for col in numeric_cols:
        col_lower = col.lower()
        
        # Provera da li izgleda kao vreme (monotono rastuće)
        if df[col].is_monotonic_increasing:
            print(f"   ✓ {col}: Monotono rastuća (dobar kandidat za vreme)")
            time_col_candidates.append((col, 'monotonic'))
        elif 'time' in col_lower or 'date' in col_lower or 't_' in col_lower:
            print(f"   ⚠️  {col}: Ime sugerije vreme ali nije monotono")
            time_col_candidates.append((col, 'name_match'))
    
    # Ako nema monotone, uzmi prvu numeričku
    if not time_col_candidates and len(numeric_cols) >= 2:
        time_col = numeric_cols[0]
        value_col = numeric_cols[1]
        print(f"\n🎯 Preporuka (automatski izbor):")
        print(f"   Vreme (t): {time_col}")
        print(f"   Vrednost (y): {value_col}")
        
        # Kreiraj testni plot
        try:
            plot_sample_data(df, time_col, value_col)
        except:
            pass
    
    return numeric_cols


def plot_sample_data(df, time_col, value_col, sample_size=1000):
    """
    Kreira jednostavan plot za pregled podataka.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Uzmi uzorak ako je dataset prevelik
        if len(df) > sample_size:
            plot_df = df.sample(sample_size).sort_values(time_col)
        else:
            plot_df = df.sort_values(time_col)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(plot_df[time_col], plot_df[value_col], s=1, alpha=0.5)
        plt.xlabel(time_col)
        plt.ylabel(value_col)
        plt.title(f'Rasipanje: {value_col} vs {time_col}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(plot_df[value_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel(value_col)
        plt.ylabel('Frekvencija')
        plt.title(f'Distribucija: {value_col}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_preview.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   📊 Plot sačuvan kao 'data_preview.png'")
        
    except Exception as e:
        print(f"   ⚠️  Nije moguće kreirati plot: {e}")


def run_wwz_test(df):
    """
    Pokreće test WWZ na učitanim podacima sa memory safe opcijom.
    
    OVA FUNKCIJA JE AŽURIRANA da koristi wwt_memory_safe
    """
    if not LIBWWZ_AVAILABLE:
        print("❌ libwwz nije instaliran. Instaliraj sa: pip install libwwz")
        return
    
    print("\n" + "=" * 80)
    print("🧪 POKRETANJE TEST WWZ SA MEMORY SAFE OPTIMIZACIJOM")
    print("=" * 80)
    
    # Prikaži numeričke kolone
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Dostupne numeričke kolone: {numeric_cols}")
    
    if len(numeric_cols) < 2:
        print("❌ Nema dovoljno numeričkih kolona za WWZ")
        return
    
    # Izbor kolona
    print("\nIzaberi kolone za WWZ analizu:")
    for i, col in enumerate(numeric_cols, 1):
        print(f"  {i}. {col}")
    
    try:
        time_idx = int(input(f"\nUnesi broj za VREMENSKU kolonu (1-{len(numeric_cols)}): ")) - 1
        value_idx = int(input(f"Unesi broj za VREDNOST kolonu (1-{len(numeric_cols)}): ")) - 1
        
        if time_idx == value_idx:
            print("❌ Vremenska i vrednosna kolona ne mogu biti iste!")
            return
            
        time_col = numeric_cols[time_idx]
        value_col = numeric_cols[value_idx]
        
        print(f"\n🎯 Izabrane kolone:")
        print(f"   Vreme (t): {time_col}")
        print(f"   Vrednost (y): {value_col}")
        
        # Priprema podataka
        clean_df = df[[time_col, value_col]].dropna()
        
        if len(clean_df) < 10:
            print("❌ Previše nedostajućih podataka!")
            return
        
        print(f"   Dostupnih podataka: {len(clean_df):,} redova")
        
        # Kreiranje frekvencija
        t_min = clean_df[time_col].min()
        t_max = clean_df[time_col].max()
        t_range = t_max - t_min
        
        print(f"   Vremenski opseg: {t_min:.2f} do {t_max:.2f} (range: {t_range:.2f})")
        
        # Automatsko određivanje frekvencija
        # Smanji broj frekvencija za memory safe
        n_freq_points = 50  # Umesto 100, koristi 50
        if len(clean_df) > 10000:
            n_freq_points = 30  # Za veće dataset-e, još manje
        
        # Proba da izračuna Nyquist ako ima dovoljno podataka
        try:
            if len(clean_df) > 100:
                sorted_t = np.sort(clean_df[time_col].values)
                time_diffs = np.diff(sorted_t)
                median_diff = np.median(time_diffs[time_diffs > 0])
                
                if median_diff > 0:
                    nyquist = 0.5 / median_diff
                else:
                    nyquist = 1.0
            else:
                nyquist = 1.0
        except:
            nyquist = 1.0
        
        freq_low = 0.1 / t_range if t_range > 0 else 0.01
        freq_high = min(nyquist * 0.8, 5.0)  # 80% Nyquista, max 5 Hz (smanjeno sa 10)
        
        freq = np.linspace(freq_low, freq_high, n_freq_points)
        
        print(f"   Frekvencije: {len(freq)} tačaka od {freq_low:.4f} do {freq_high:.4f} Hz")
        
        # SMANJI TAU TAČKE - OVO JE KLJUČNO ZA MEMORIJU!
        ntime = len(clean_df)
        if ntime > 1000:
            # Koristi manji broj uniformnih tau tačaka
            ntau_target = min(500, ntime // 20)  # Max 500, ili 5% od ntime
            tau = np.linspace(t_min, t_max, ntau_target)
            print(f"\n🎯 Optimizacija memorije:")
            print(f"   Originalne vremenske tačke: {ntime:,}")
            print(f"   Tau tačaka (smanjeno): {len(tau):,}")
            print(f"   Redukcija: {ntime/len(tau):.1f}x manje memorije")
        else:
            tau = None  # Koristi default (sve vremenske tačke)
            print(f"\nℹ️  Dataset mali ({ntime} tačaka), koristim sve tau tačke")
        
        # Izbor metode
        print("\n🔧 IZBOR METODE WWZ:")
        print("  1. Memory Safe + GPU (preporučeno)")
        print("  2. Memory Safe + CPU")
        print("  3. Standard WWZ (može pući na velikim podacima)")
        
        method_choice = input("\nIzaberi metodu (1-3, default=1): ").strip()
        
        if method_choice == "2":
            use_memory_safe = True
            use_gpu = False
            method_name = "Memory Safe (CPU)"
        elif method_choice == "3":
            use_memory_safe = False
            use_gpu = GPU_AVAILABLE
            method_name = f"Standard ({'GPU' if GPU_AVAILABLE else 'CPU'})"
        else:  # default ili "1"
            use_memory_safe = True
            use_gpu = GPU_AVAILABLE
            method_name = f"Memory Safe ({'GPU' if GPU_AVAILABLE else 'CPU'})"
        
        # Pokretanje WWZ
        response = input(f"\n🚀 Pokrenuti {method_name} WWZ analizu? (da/ne): ").strip().lower()
        
        if response == 'da':
            print(f"\n⏳ Izvršavam {method_name}...")
            start_time = time.time()
            
            try:
                if use_memory_safe:
                    # KORISTI MEMORY SAFE VERZIJU
                    result = wwt_memory_safe(
                        clean_df[time_col].values,
                        clean_df[value_col].values,
                        freq,
                        tau=tau,
                        use_gpu=use_gpu,
                        max_memory_gb=6.0,  # Max 6GB za 8GB GPU
                        tau_downsample=True
                    )
                else:
                    # KORISTI STANDARDNU VERZIJU
                    if use_gpu and GPU_AVAILABLE:
                        result = wwt_gpu(
                            clean_df[time_col].values,
                            clean_df[value_col].values,
                            freq,
                            tau=tau,
                            gpu_chunk_size=1,  # Samo 1 frekvencija odjednom
                            optimize_memory=True
                        )
                    else:
                        result = wwt_cpu(
                            clean_df[time_col].values,
                            clean_df[value_col].values,
                            freq,
                            tau=tau
                        )
                
                elapsed = time.time() - start_time
                
                if result:
                    wwz, dampp = result
                    print(f"\n✅ WWZ ZAVRŠEN! ({method_name})")
                    print(f"   Vreme izvršenja: {elapsed:.2f} sekundi")
                    print(f"   Rezultat shape: {wwz.shape}")
                    print(f"   WWZ min/max: {wwz.min():.3f} / {wwz.max():.3f}")
                    
                    # Kreiraj jednostavan plot
                    try:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(12, 8))
                        
                        # Ako imamo tau, koristi ga za x-osu
                        if tau is not None:
                            x_axis = tau
                        else:
                            x_axis = clean_df[time_col].values
                        
                        plt.subplot(2, 1, 1)
                        plt.scatter(clean_df[time_col].values, clean_df[value_col].values, 
                                   s=1, alpha=0.5, c='blue')
                        plt.xlabel(f'Time ({time_col})')
                        plt.ylabel(f'Value ({value_col})')
                        plt.title(f'Original Data: {value_col} vs {time_col}')
                        plt.grid(True, alpha=0.3)
                        
                        plt.subplot(2, 1, 2)
                        plt.imshow(wwz, aspect='auto', 
                                  extent=[x_axis.min(), x_axis.max(), freq_low, freq_high],
                                  origin='lower', cmap='viridis')
                        plt.colorbar(label='WWZ Power')
                        plt.xlabel(f'Time ({time_col})')
                        plt.ylabel('Frequency (Hz)')
                        plt.title(f'WWZ Analysis ({method_name}, {elapsed:.1f}s)')
                        
                        plt.tight_layout()
                        
                        # Sačuvaj plot sa imenima kolona
                        safe_time_col = time_col.replace('/', '_').replace(' ', '_')
                        safe_value_col = value_col.replace('/', '_').replace(' ', '_')
                        plot_filename = f'wwz_{safe_time_col}_{safe_value_col}.png'
                        
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        print(f"   📊 Plot sačuvan kao '{plot_filename}'")
                        
                        # Sačuvaj i rezultate u NPZ fajl
                        results_filename = f'wwz_results_{safe_time_col}_{safe_value_col}.npz'
                        np.savez(results_filename, 
                                 wwz=wwz, 
                                 dampp=dampp, 
                                 freq=freq, 
                                 time_points=x_axis,
                                 time_col=time_col,
                                 value_col=value_col)
                        print(f"   💾 Rezultati sačuvani kao '{results_filename}'")
                        
                        plt.show()
                        
                    except Exception as e:
                        print(f"   ⚠️  Nije moguće kreirati plot: {e}")
                        
                        # Ipak sačuvaj rezultate
                        results_filename = 'wwz_results_backup.npz'
                        np.savez(results_filename, wwz=wwz, dampp=dampp, freq=freq)
                        print(f"   💾 Rezultati sačuvani kao '{results_filename}'")
                
            except MemoryError as e:
                print(f"\n❌ OUT OF MEMORY čak i sa optimizacijom!")
                print(f"   Greška: {e}")
                
                # LAST RESORT: Koristi CPU i još manje tau
                print("\n🔄 Pokušavam sa CPU i ekstremnom redukcijom...")
                
                # Ekstremno smanji
                tau_tiny = np.linspace(t_min, t_max, 100)  # Samo 100 tau tačaka
                freq_tiny = freq[:25]  # Samo 25 frekvencija
                
                start = time.time()
                try:
                    wwz, damp = wwt_cpu(clean_df[time_col].values, clean_df[value_col].values, 
                                       freq_tiny, tau=tau_tiny)
                    elapsed = time.time() - start
                    
                    print(f"   CPU uspešno sa redukovanim podacima!")
                    print(f"   Vreme: {elapsed:.2f}s")
                    print(f"   Shape: {wwz.shape}")
                except:
                    print("   ❌ Ne mogu ni sa ekstremnom redukcijom.")
                    print("   Probaj da smanjiš dataset pre pokretanja.")
            
            except Exception as e:
                print(f"\n❌ Greška pri WWZ: {e}")
                import traceback
                traceback.print_exc()
            
    except (ValueError, IndexError) as e:
        print(f"❌ Nevalidan unos: {e}")
    except Exception as e:
        print(f"❌ Greška: {e}")


def main():
    """
    Glavna funkcija za pokretanje analize.
    """
    # Unesi putanju do Parquet fajla
    file_path = input("Unesi putanju do Parquet fajla: ").strip()
    
    # Ili koristi default ako nije uneseno
    if not file_path:
        # Traži parquet fajlove u trenutnom direktorijumu
        parquet_files = list(Path('.').glob('*.parquet'))
        
        if parquet_files:
            file_path = str(parquet_files[0])
            print(f"\n🔍 Pronađen Parquet fajl: {file_path}")
        else:
            print("\n❌ Nema Parquet fajlova u trenutnom direktorijumu.")
            print("   Kopiraj Parquet fajl u ovaj direktorijum ili unesi punu putanju.")
            return
    
    # Pokreni analizu
    df = explore_parquet_file(file_path)
    
    # Daj preporuke za WWZ
    if df is not None:
        suggest_wwz_columns(df)
        
        # Opcija za pokretanje WWZ
        if LIBWWZ_AVAILABLE:
            response = input("\n🧪 Želiš li da pokreneš test WWZ na ovim podacima? (da/ne): ").strip().lower()
            
            if response == 'da':
                run_wwz_test(df)
        else:
            print("\n⚠️  libwwz nije instaliran, WWZ analiza nije dostupna.")


if __name__ == "__main__":
    main()
