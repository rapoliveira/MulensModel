#!/bin/bash

# This script is still being improved...
# Usage: ./run_tasks.sh <task>
# <task>: Specify the task to run:
#         - task1: Fit 2L1S models with EMCEE for multiple events.
#         - task2: Fit 1L2S models with UltraNest for multiple events.
#         - task3: Fit 2L1S models with UltraNest for multiple events.

if [ $# -eq 0 ]; then
    echo "No arguments provided. Options are: task1, task2, or task3."
    exit 1
fi

ARG=$1
python_script="../examples/example_16/ulens_model_fit.py"
project_name="OGLE-evfinder"

if [ "$ARG" == "task1" ]; then
    DIRECTORY="./$project_name/yaml_files_2L1S"
    yaml_files=$(ls "$DIRECTORY"/*.yaml | sort)
elif [ "$ARG" == "task2" ]; then
    DIRECTORY="./$project_name/ultranest_1L2S"
    yaml_files=$(ls "$DIRECTORY"/*-1L2S_UltraNest.yaml | sort)
elif [ "$ARG" == "task3" ]; then
    DIRECTORY="./$project_name/ultranest_2L1S"
    yaml_files=$(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort)
else
    echo "Invalid argument. Please use: task1, task2, or task3."
    exit 1
fi

for yaml_file in $yaml_files
do
    echo -e "\n--\n\nProcessing $yaml_file"
    python3 "$python_script" "$yaml_file"
done

# for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort | head -n 7)
# for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort | head -n 15 | tail -n 2)
# for file in $(ls "$DIRECTORY"/*-2L1S_UltraNest.yaml | sort | head -n 23 | tail -n 8)