import matplotlib.pyplot as plt
import sys

def plot_benchmark(filename="results.csv"):
    data = {}

    try:
        with open(filename, "r") as f:
            next(f)  # Skip header
            
            for line in f:
                if not line.strip(): continue                
                parts = line.strip().split(",")
                kernel = parts[0]
                N = int(parts[1])
                time = float(parts[2])

                if kernel not in data:
                    data[kernel] = []
                
                data[kernel].append((N, time))

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    plt.figure(figsize=(10, 6))
    
    timesteps = 2000

    for kernel, points in data.items():
        points.sort(key=lambda x: x[0])
        
        N_list = [p[0] for p in points]
        norm_time_list = [p[1] / (p[0]**3) / timesteps for p in points]

        if "pml" in kernel or "warp" in kernel:
            current_linestyle = '-' 
        else:
            current_linestyle = ':'

        if "fdtdx" in kernel:
            current_color = 'tab:red'
        elif "meep" in kernel:
            current_color = 'tab:blue'
        elif "warp" in kernel:
            current_color = 'darkgreen'
        else:
            current_color = 'limegreen'

        plt.plot(
            N_list, 
            norm_time_list, 
            linestyle=current_linestyle, 
            color=current_color,
            label=kernel,
            marker='o',      
            markersize=3,    
            linewidth=1.0   
        )

    plt.yscale('log')
    plt.xlabel("Grid Size ($N$)")
    plt.ylabel(r"Normalized Time ($t / N^3$)")
    plt.title("FDTD Performance Scaling")
    plt.legend(title="Kernel")
    
    plt.grid(True, which="both", ls="-", alpha=0.4, linewidth=0.5)
    plt.tight_layout()
    
    output_file = "benchmark_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    plot_benchmark(fname)
