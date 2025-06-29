import esim_py
import os
import numpy as np
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate event data from grayscale PNG")
    parser.add_argument('--file', type=str, required=True, help="Path to the folder containing on_centre or off_centre data")
    args = parser.parse_args()



    # Input and output directories
    file = args.file
    input_folderIm = r"im\gray"
    input_time = "Timestamps\Timestamps.txt"
    output_name = "event\event.bin"

    input_folderIm = os.path.join(file, input_folderIm)
    timestamp_path = os.path.join(file, input_time)
    output_event_file = os.path.join(file, output_name)
    # Reasonable values for ESIM constructor parameters
    contrast_threshold_pos = 0.2  # Positive contrast threshold
    contrast_threshold_neg = 0.2  # Negative contrast threshold
    refractory_period = 1e-3  # 1 ms refractory period
    log_eps = 1e-3  # Small epsilon for numerical stability
    use_log = True  # Use logarithmic intensity

    # Initialize the EventSimulator
    esim = esim_py.EventSimulator(
        contrast_threshold_pos,  # Contrast threshold for positive events
        contrast_threshold_neg,  # Contrast threshold for negative events
        refractory_period,  # Refractory period in seconds
        log_eps,  # Epsilon for log stability
        use_log  # Whether or not to use log intensity
    )
    print("Initialized")

    with open(timestamp_path, 'r') as f:
        for line in f:
            try:
                float(line.strip())  # Try converting each line to a float
            except ValueError:
                print(f"Invalid timestamp found: {line}")

    # Generate events from the video and corresponding timestamps
    events_from_image = esim.generateFromFolder(
        input_folderIm,  # Path to the input video file
        timestamp_path  # Path to the timestamps file
    )
    print("generated events")
    # Save events with batch writing

    batch_size = 1000000  # Number of events to accumulate in memory before writing to disk

    # Tqdm to show progress bar
    total_events = len(events_from_image)  # Total number of events to process

    with open(output_event_file, 'wb') as f:
        event_buffer = []
        for event in tqdm(events_from_image, total=total_events, desc="Writing events to file"):
            # event is in the form [x, y, timestamp, polarity], cast it to a numpy array
            event_array = np.array(event, dtype=np.float32)
            event_buffer.append(event_array)

            # Write the batch to file once the batch size is reached
            if len(event_buffer) >= batch_size:
                np.save(f, np.array(event_buffer))  # Write in binary format
                event_buffer = []  # Clear buffer

        # Write any remaining events in the buffer
        if event_buffer:
            np.save(f, np.array(event_buffer))

    print(f"Events generated and saved to {output_event_file}")

if __name__ == "__main__":
    main()