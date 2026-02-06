import pandas as pd
import numpy as np
import sys

def generate_final_approximations(input_file="results.csv", output_file="approx.csv"):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    df = df[df['N'] > 119]

    if 'time' not in df.columns and 'Runtime_Seconds' in df.columns:
        df = df.rename(columns={'Runtime_Seconds': 'time'})
        
    df_pivot = df.pivot(index='N', columns='Kernel', values='time')
    
    if 'yee' not in df_pivot.columns or 'pml' not in df_pivot.columns:
        print("Error: Input CSV must contain 'yee' and 'pml' measurements.")
        return

    N = df_pivot.index.to_numpy()
    T_yee_measured = df_pivot['yee'].to_numpy()
    T_pml_measured = df_pivot['pml'].to_numpy()

    UNITS_YEE_ANISO = 24.0
    UNITS_PML_ANISO = 48.0
    
    UNITS_YEE_ISO = 16.0
    UNITS_PML_ISO = 40.0

    FACTOR_YEE_ISO = UNITS_YEE_ISO / UNITS_YEE_ANISO 
    FACTOR_PML_ISO = UNITS_PML_ISO / UNITS_PML_ANISO 

    
    T_yee_iso = T_yee_measured * FACTOR_YEE_ISO
    T_pml_iso = T_pml_measured * FACTOR_PML_ISO

    
    PML_THICKNESS = 20
    dim_core = np.maximum(0, N - 2 * PML_THICKNESS)
    vol_core = dim_core ** 3
    vol_total = N ** 3
    
    frac_core = vol_core / vol_total
    frac_pml = 1.0 - frac_core

    T_split_aniso = (frac_core * T_yee_measured) + (frac_pml * T_pml_measured)
    
    T_split_iso = (frac_core * T_yee_iso) + (frac_pml * T_pml_iso)

    dfs = []
    
    dfs.append(pd.DataFrame({
        'Kernel': ['split_approx'] * len(N),
        'N': N,
        'time': T_split_aniso
    }))
    
    dfs.append(pd.DataFrame({
        'Kernel': ['yee_isotrop_approx'] * len(N),
        'N': N,
        'time': T_yee_iso
    }))
    
    dfs.append(pd.DataFrame({
        'Kernel': ['pml_isotrop_approx'] * len(N),
        'N': N,
        'time': T_pml_iso
    }))
    
    dfs.append(pd.DataFrame({
        'Kernel': ['split_isotrop_approx'] * len(N),
        'N': N,
        'time': T_split_iso
    }))
    
    final_df = pd.concat(dfs, ignore_index=True)
    
    final_df = final_df.sort_values(by=['Kernel', 'N'])

    final_df.to_csv(output_file, index=False)
    print(f"Approximation results saved to {output_file}")
    
    idx = -1 
    n_max = N[idx]
    
    print(f"\n--- Performance Model Summary (N={n_max}) ---")
    print(f"MEASURED BASELINES:")
    print(f"  Yee (Aniso):          {T_yee_measured[idx]:.2f} s")
    print(f"  PML (Aniso):          {T_pml_measured[idx]:.2f} s")
    print("-" * 40)
    print(f"APPROXIMATIONS:")
    print(f"  1. Split (Current):   {T_split_aniso[idx]:.2f} s  (Weighted Avg of Measured)")
    print(f"  2. Yee (Isotrop):     {T_yee_iso[idx]:.2f} s  (-33% vs Yee Aniso)")
    print(f"  3. PML (Isotrop):     {T_pml_iso[idx]:.2f} s  (-17% vs PML Aniso)")
    print(f"  4. Split (Isotrop):   {T_split_iso[idx]:.2f} s  (Weighted Avg of Isotrop)")

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "approx.csv"
    generate_final_approximations(infile, outfile)
