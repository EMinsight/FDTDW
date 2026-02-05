#!/bin/bash

OUTPUT_FILE="results_meep.csv"
echo "Kernel,N,time" > "$OUTPUT_FILE"

SIZES="41 $(seq 48 8 128) $(seq 160 32 448)"

MODES=(0 1)

for N in $SIZES; do
    for mode in "${MODES[@]}"; do
        
        if [ "$mode" -eq 1 ]; then
            LABEL="meep_cpu_mpi_pml"
            echo "Running Meep (MPI) | N=$N | PML=ON"
        else
            LABEL="meep_cpu_mpi_periodic"
            echo "Running Meep (MPI) | N=$N | PML=OFF"
        fi
        
        output=$(mpirun -np 12 python meep0.py --N "$N" --pml "$mode" 2>&1)
        
        runtime=$(echo "$output" | grep "Elapsed run time =" | awk '{print $5}')
        
        if [ -z "$runtime" ]; then
            echo "Error: Could not parse runtime for N=$N Mode=$mode"
        else
            echo "  -> Time: $runtime s"
            echo "$LABEL,$N,$runtime" >> "$OUTPUT_FILE"
        fi
        
    done
done

echo "------------------------------------------------"
echo "Benchmark Complete. Results saved to $OUTPUT_FILE"
