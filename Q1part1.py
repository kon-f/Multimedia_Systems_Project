import cv2
import matplotlib.pyplot as plt  #only for "commented" code
import numpy as np
import sys

# Read the video
video = 'ball.mp4'
cap = cv2.VideoCapture(video)


# Initialize variables
frame_count = 0
frames = []

# Read and extract frames from the video
while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()

    # If the frame was read successfully
    if ret:
        # Append the frame to the list
        frames.append(frame)
        
    else:
        break

    # Increment the frame count
    frame_count += 1

# Release the video capture object
cap.release()

# Separate I-frame and P-frames
i_frame = frames[0]

# Print the total number of frames extracted
print("Total frames extracted:", frame_count)

# Calculate and store image error frames
error_frames = []
for i in range(1, len(frames)):
    error_frame = cv2.absdiff(frames[i], frames[i-1])
    error_frames.append(error_frame)

print("Error frame sequence created")
    
# Visualize the image error sequence
for error_frame in error_frames:
    # Display the error frame
    cv2.imshow('Error Frame', error_frame)
    
    # Add a small delay to display the frames at a reasonable speed, here it's 25 milliseconds
    cv2.waitKey(25)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()
    
# Get the frame shape from the first frame. Will be needed for decoding
frame_shape = error_frames[0].shape
    
# Convert error_frames to bytes
error_frames_bytes = [frame.tobytes() for frame in error_frames]

# Perform run-length encoding on the error_frames bytes
compressed_data = []
for frame_bytes in error_frames_bytes:
    compressed_frame = bytearray()
    count = 1
    for i in range(1, len(frame_bytes)):
        value = frame_bytes[i-1]
        if not 0 <= value <= 255:
            print("Error: Byte value outside valid range")
            print("Problematic byte value:", value)
            break
        if frame_bytes[i] == value and count < 255:
            count += 1
        else:
            compressed_frame.extend([value, count])
            count = 1
    else:
        value = frame_bytes[-1]
        if not 0 <= value <= 255:
            print("Error: Byte value outside valid range")
            print("Problematic byte value:", value)
            break
        compressed_frame.extend([value, count])
        compressed_data.append(compressed_frame)

print("Error frame sequence compressed using lossless Run length encoding")

# Decode the run-length encoded data
decoded_frames = []
for compressed_frame in compressed_data:
    decoded_frame = bytearray()
    for i in range(0, len(compressed_frame), 2):
        value = compressed_frame[i] 
        count = compressed_frame[i+1]
        decoded_frame.extend([value] * count)
    decoded_frames.append(np.frombuffer(decoded_frame, dtype=np.uint8).reshape(frame_shape))

print("Error frame sequence decompressed")


# # Display the decoded frames
# for decoded_frame in decoded_frames:
#     cv2.imshow('Decoded Frame', decoded_frame)
#     cv2.waitKey(25)  # Add a small delay to display the frames at a reasonable speed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break    

# # Display the decoded frames with matplotlib.pyplot
# for decoded_frame in decoded_frames:
#     plt.imshow(decoded_frame, cmap='gray')
#     plt.pause(0.001)  # Add a small delay to allow the figure to update
#     plt.show()


# Compare the pre-compressed error sequence with the decompressed sequence
equal = True
for i in range(len(error_frames)):

    # Check if both frames are identical element-wise
    if not np.array_equal(error_frames[i], decoded_frames[i]):
        print(f"Frame {i}: Compression is NOT lossless")
        equal = False
        
if equal:
    print("Pre-compressed and decompressed frames are equal: Lossless compression")

# Calculate the total size of the image data in error_frames and compressed_data
size_frames_data = sum(sys.getsizeof(frame.tobytes()) for frame in frames)
size_error_frames_data = sum(sys.getsizeof(frame) for frame in error_frames_bytes)
size_compressed_data = sum(sys.getsizeof(frame) for frame in compressed_data)

# Print the sizes and compression ratio
print(f"Size of frames before error frame sequence creation: {size_frames_data} bytes")
print(f"Size of error_frames: {size_error_frames_data} bytes")
print(f"Size of compressed_data: {size_compressed_data} bytes")
print(f"Compression Ratio of error sequence: {size_error_frames_data / size_compressed_data:.2f}")
print(f"Total Compression Ratio: {size_frames_data / size_compressed_data:.2f}")

# Close all OpenCV windows
cv2.destroyAllWindows()

