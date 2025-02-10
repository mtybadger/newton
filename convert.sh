#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <n>"
    echo "Converts frames from video_0 through video_n folders to MP4s"
    exit 1
fi

n=$1

# Check if n is a valid non-negative integer
if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "Error: n must be a non-negative integer"
    exit 1
fi

# Loop through each video folder
for i in $(seq 0 $n); do
    input_folder="video_$i"
    output_file="output_$i.mp4"
    
    # Check if input folder exists
    if [ ! -d "$input_folder" ]; then
        echo "Warning: Folder $input_folder not found, skipping..."
        continue
    fi
    
    echo "Converting $input_folder to $output_file..."
    
    # Use ffmpeg to convert the sequence of JPGs to MP4
    ffmpeg -y -framerate 24 -i "$input_folder/frame_%04d.jpg" \
        -c:v libx264 -pix_fmt yuv420p -crf 23 "$output_file"
        
    if [ $? -eq 0 ]; then
        echo "Successfully created $output_file"
    else
        echo "Error converting $input_folder"
    fi
done

echo "Conversion complete!"

