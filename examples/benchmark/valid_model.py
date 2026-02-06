import pandas as pd
import numpy as np

# Hardware Specs (RTX 4070 Ti Super)
GPU_BW_GBs = 672.0       # Theoretical Max Memory Bandwidth (GB/s)
GPU_TFLOPS = 44.0        # Theoretical Max FP32 Compute (TFLOPS)

#   N_fields:     Ex, Ey, Ez, Hx, Hy, Hz (6 total)
#   N_yeecoeff:   CEA, CEB... (12 total)
#   N_auxfields:  psi_ex_y... (12 total)
#   N_pmlcoeff:   B_E_X... (Ignored if 1D cached, set >0 if 3D)

N_fields = 6.0
N_yeecoeff = 12.0
N_auxfields = 12.0
N_pmlcoeff = 0.0

# Each field component is updated once (RW = 2 ops) and read as neighbor once (Read = 1 op)
# Total Field Traffic = (2 + 1) * N_fields = 3 * N_fields
# Coefficients are read once = 1 * N_coeffs
# Aux fields are RW = 2 * N_auxfields

Floats_Yee = (3 * N_fields) + N_yeecoeff 
Floats_Full = Floats_Yee + (2 * N_auxfields) + N_pmlcoeff

Bytes_Yee = Floats_Yee * 4
Bytes_Full = Floats_Full * 4

FLOPS_YEE = 36.0
FLOPS_FULL = 96.0

TIMESTEPS = 2000

print(f"--- Model Parameters ---")
print(f"Yee Payload: {Bytes_Yee:.0f} Bytes/cell ({Floats_Yee:.0f} Floats)")
print(f"Full Payload: {Bytes_Full:.0f} Bytes/cell ({Floats_Full:.0f} Floats)")
print(f"Target Mem Ratio: {Bytes_Full/Bytes_Yee:.2f}")
print(f"Target Flop Ratio: {FLOPS_FULL/FLOPS_YEE:.2f}")

try:
    df = pd.read_csv("results.csv")
    df.columns = df.columns.str.strip()
    if 'time' in df.columns and 'Runtime_Seconds' not in df.columns:
        df['Runtime_Seconds'] = df['time']
    elif 'Runtime_Seconds' in df.columns and 'time' not in df.columns:
        df['time'] = df['Runtime_Seconds']
        
except Exception as e:
    print(f"Error reading results.csv: {e}")
    exit()

max_N = df['N'].max()
df_max = df[df['N'] == max_N]

try:
    t_yee = df_max[df_max['Kernel'] == 'yee']['time'].values[0]
    t_full = df_max[df_max['Kernel'] == 'pml']['time'].values[0] 
    t_warp = df_max[df_max['Kernel'] == 'warp']['time'].values[0] 
except IndexError:
    print(f"Error: Missing kernel data at N={max_N}")
    exit()

print(f"\n--- Measured Results (N={max_N}) ---")
print(f"Time Yee:  {t_yee:.4f} s")
print(f"Time Full: {t_full:.4f} s")
print(f"Time Warp: {t_warp:.4f} s")

ratio_time = t_full / t_yee
ratio_mem = Bytes_Full / Bytes_Yee
ratio_flop = FLOPS_FULL / FLOPS_YEE

print(f"\n--- Validation ---")
print(f"Measured Time Ratio: {ratio_time:.2f}")
print(f"Predicted Mem Ratio: {ratio_mem:.2f}  <-- Diff: {abs(ratio_time-ratio_mem):.2f}")
print(f"Predicted Flop Ratio:{ratio_flop:.2f}  <-- Diff: {abs(ratio_time-ratio_flop):.2f}")

if abs(ratio_time - ratio_mem) < abs(ratio_time - ratio_flop):
    print(">> CONCLUSION: Solver is MEMORY BOUND")
else:
    print(">> CONCLUSION: Solver is COMPUTE BOUND")

vol_total = (max_N**3) * TIMESTEPS

bw_yee = (vol_total * Bytes_Yee) / t_yee / 1e9
bw_full = (vol_total * Bytes_Full) / t_full / 1e9

print(f"\n--- Bandwidth (vs 4070 Ti Super {GPU_BW_GBs} GB/s) ---")
print(f"Yee:  {bw_yee:.0f} GB/s ({bw_yee/GPU_BW_GBs*100:.1f}%)")
print(f"Full: {bw_full:.0f} GB/s ({bw_full/GPU_BW_GBs*100:.1f}%)")

t_compute_ideal_yee = (vol_total * FLOPS_YEE) / (GPU_TFLOPS * 1e12)

print(f"\n--- Compute (vs 4070 Ti Super {GPU_TFLOPS} TFLOPS) ---")
print(f"Ideal Compute Time (Yee):  {t_compute_ideal_yee:.5f} s")
print(f"Actual Time (Yee):         {t_yee:.4f} s")
print(f"Slowdown Factor:           {t_yee/t_compute_ideal_yee:.1f}x (Due to Memory Bottleneck)")
