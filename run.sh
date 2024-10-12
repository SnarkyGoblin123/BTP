#!/bin/bash

# Set variables
PYTHON_SCRIPT="packing.py"  # Replace with the name of your Python script
DEST_DIR="/home/joel/gps-sdr-sim-master/gen_data_split"  # Replace with your destination folder path
FINAL_DEST_DIR="/home/joel/BTP/data"  # Replace with final destination folder path
CONFIG_FILE="./test.conf"
COMBINED_OUTPUT="combined_data.dat"

source ../Documents/env_fold/python3.10/bin/activate

# Define the range of 'k' values (e.g., from 0.01 to 0.05 with a step of 0.01)
for k in $(seq -2 1 3); do
    # Run the Python script with 'k' as an argument
    python3 $PYTHON_SCRIPT $k

    # Copy the output file to the destination folder
    OUTPUT_FILE="data/output_upd.dat"
    cp $OUTPUT_FILE $DEST_DIR

    # Change directory to the destination folder
    cd $DEST_DIR

    # Choose the file to combine with the newly generated one (this could be a fixed file or other files)
    cat output_upd.dat mydatab mydatac mydatad mydatae mydataf > $COMBINED_OUTPUT

    # Move the combined output to the final destination folder
    mv $COMBINED_OUTPUT $FINAL_DEST_DIR/combined_data.dat
    
    cd /home/joel/BTP
    # sudo chmod 777 combined_data.dat
    # cd ../

    # Run the GNSS-SDR tool with the specified config file and save the log output
    gnss-sdr --config_file=$CONFIG_FILE > log_k_${k}.txt

    # Completion message
    echo "Process completed for k=$k. Combined data file moved to $FINAL_DEST_DIR/combined_data_k_${k}.dat. Log saved as log_k_${k}.txt"

    # Optionally, you can clean up the individual output file to save space
    # rm $OUTPUT_FILE

done
