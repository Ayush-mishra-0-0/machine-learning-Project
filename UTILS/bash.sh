#!/bin/bash

OUTPUT_DIR="processed_data/FGVC8"
TRAIN_DIR="${OUTPUT_DIR}/train"
VAL_DIR="${OUTPUT_DIR}/val"
TEST_DIR="${OUTPUT_DIR}/test"

mkdir -p "$VAL_DIR" "$TEST_DIR"

move_folders() {
    local source_dir=$1
    local dest_dir=$2
    local num_to_move=$3
    local description=$4

    echo "Moving folders to $description..."
    find "$source_dir" -maxdepth 1 -mindepth 1 -type d | shuf -n "$num_to_move" | while read folder; do
        mv "$folder" "$dest_dir"
        echo "Moved $(basename "$folder") to $description"
    done
}

total_folders=$(find "$TRAIN_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)

move_count=$((total_folders / 10))

move_folders "$TRAIN_DIR" "$VAL_DIR" "$move_count" "validation"

total_folders=$(find "$TRAIN_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
move_count=$((total_folders / 10))

move_folders "$TRAIN_DIR" "$TEST_DIR" "$move_count" "test"

echo "Splitting completed."
echo "Folders in val: $(find "$VAL_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "Folders in test: $(find "$TEST_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "Folders remaining in train: $(find "$TRAIN_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)"


# Folders in val: 855
# Folders in test: 770
# Folders remaining in train: 6934