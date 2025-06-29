import numpy as np
import os
import argparse



def create_timestepts(file_out, file_name, n_frames=480, duration=4):
    # create time stamps
    time_stamps = np.linspace(0, duration, n_frames)

    # Construct the full file path
    file_path = os.path.join(file_out, file_name)

    # Write the time stamps to the file
    with open(file_path, 'w') as f:
        for ts in time_stamps:
            f.write(f"{ts}\n")

    print(f"Time stamps written to {file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create timestamps for ")
    parser.add_argument('--file', type=str, required=True, help="Path to the folder containing on_centre or off_centre data")
    parser.add_argument("--n_frames", type=float, default=480, help="Number of frames (default 480)")
    parser.add_argument("--duration", type=float, default=4, help="Duration of sequence (default: 4s)")

    args = parser.parse_args()

    file_out = args.file
    n_frames = args.n_frames
    duration = args.duration

    file_name = r"Timestamps\Timestamps.txt"
    create_timestepts(file_out, file_name, n_frames, duration)