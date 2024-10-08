#!/bin/bash

# This script is still being improved...
# Usage: ./run_tasks.sh <task>
# <task>: Specify the task to run:
#         - task1: Fit 2L1S models with EMCEE for multiple events.
#         - task2: Fit 1L2S models with UltraNest for multiple events.
#         - task3: Fit 2L1S models with UltraNest for multiple events.

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Options are: task1, task2, or task3."
    exit 1
fi

ARG=$1

if [ "$ARG" == "task1" ]; then
    DIRECTORY="./OGLE-evfinder/yaml_files_2L1S"
    for file in $(ls "$DIRECTORY"/*.yaml | sort)
    do
        echo -e "\n--\n\nProcessing $file"
        python3 ../examples/example_16/ulens_model_fit.py "$file"
    done

elif [ "$ARG" == "task2" ]; then
    DIRECTORY="./OGLE-evfinder/ultranest_1L2S"
    for file in $(ls "$DIRECTORY"/*-1L2S_UltraNest.yaml | sort)
    do
        echo -e "\n--\n\nProcessing $file"
        python3 ../examples/example_16/ulens_model_fit.py "$file"
    done

elif [ "$ARG" == "task3" ]; then
    DIRECTORY="./OGLE-evfinder/ultranest_2L1S"
    for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort)
    do
    echo -e "\n--\n\nProcessing $file"
    python3 ../examples/example_16/ulens_model_fit.py "$file"

done

else
    echo "Invalid argument. Please use: task1, task2, or task3."
    exit 1
fi

# for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort | head -n 7)
# for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort | head -n 15 | tail -n 2)
# for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort | head -n 23 | tail -n 8)