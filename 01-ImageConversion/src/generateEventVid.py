import numpy as np
import cv2  # OpenCV for video writing
import os
import argparse

# Function to visualize events into an RGB image
def viz_events(events, resolution):
    pos_events = events[events[:, -1] == 1]  # Positive polarity events
    neg_events = events[events[:, -1] == -1]  # Negative polarity events

    image_pos = np.zeros(resolution[0] * resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0] * resolution[1], dtype="uint8")

    np.add.at(image_pos, (pos_events[:, 0] + pos_events[:, 1] * resolution[1]).astype("int32"), pos_events[:, -1] ** 2)
    np.add.at(image_neg, (neg_events[:, 0] + neg_events[:, 1] * resolution[1]).astype("int32"), neg_events[:, -1] ** 2)

    image_rgb = np.stack(
        [
            image_pos.reshape(resolution),
            image_neg.reshape(resolution),
            np.zeros(resolution, dtype="uint8")
        ], -1
    ) * 50  # Multiplying by 50 to enhance visibility

    return image_rgb


# Function to process events incrementally from the event file
def process_events_in_chunks(filepath, frame_duration, fps, output_video_path, resolution):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (resolution[1], resolution[0]))

    # Open the binary file
    with open(filepath, 'rb') as f:
        # Read the entire binary file into a numpy array (assuming float32 for x, y, timestamp, polarity)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, 4)

    current_time = data[0, 2]  # First event timestamp
    next_frame_time = current_time + frame_duration
    frame_events = []

    for event in data:
        x, y, timestamp, polarity = event

        # Check if the event belongs to the current frame
        if timestamp <= next_frame_time:
            frame_events.append([x, y, polarity])
        else:
            # Visualize and write the current frame
            if frame_events:  # Only create a frame if there are events
                image_rgb = viz_events(np.array(frame_events), resolution)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                out.write(image_bgr)

            # Move to the next frame's time window
            frame_events = [[x, y, polarity]]  # Start collecting events for the next frame
            current_time = timestamp
            next_frame_time = current_time + frame_duration

    # If there are remaining events for the last frame
    if frame_events:
        image_rgb = viz_events(np.array(frame_events), resolution)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        out.write(image_bgr)

    # Release the video writer
    out.release()
    print(f"Event video saved to {output_video_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a video from event data.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing the event data.')
    parser.add_argument('--event_file', type=str, default='event.bin', help='Name of the event data file. Default is event.bin.')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for the output video.')
    parser.add_argument('--height', type=int, default=720, help='Height of the output video.')
    parser.add_argument('--width', type=int, default=1280, help='Width of the output video.')

    args = parser.parse_args()

    input_folder = args.input_folder
    event_file = args.event_file
    fps = args.fps
    H, W = args.height, args.width

    output_video_path = os.path.join(input_folder, "event_visualization.mp4")
    frame_duration = 1.0 / fps  # Time duration of each frame in seconds

    # Process the events and generate the video
    process_events_in_chunks(os.path.join(input_folder, event_file), frame_duration, fps, output_video_path, [H, W])

if __name__ == "__main__":
    main()
    
