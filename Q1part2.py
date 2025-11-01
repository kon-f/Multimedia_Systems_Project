import cv2
import numpy as np 

def read_video_frames():
    """
    Function that reads frames from a video, converts them to grayscale, and stores them in a list.
    """
    # Read the video
    video = 'ball.mp4'
    cap = cv2.VideoCapture(video)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error opening video file")

    # Initialize variables
    frame_count = 0
    frames = []

    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()

        # If the frame was read successfully
        if ret:
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Append the grayscale frame to the list
            frames.append(gray_frame)

        else:
            break

        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()
    
    # Print the total number of frames extracted
    print("Total frames extracted:", frame_count)

    # Now you have a list of grayscale frames for further processing
    
    return frames  # Return the list of frames

# Function to predict frames and create a predicted frame list
def predict_frames(frames):
    predicted_frames = [frames[0]]  # Initial frame is the same as the first actual frame

    for i in range(1, len(frames)):
        predicted_frames.append(frames[i - 1])

    return predicted_frames

# Define the split_frame function to split a frame into blocks
def split_frame(frame, block_size=64):
    height, width = frame.shape
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = frame[y:y+block_size, x:x+block_size]
            blocks.append(block)
    return blocks

# Define the get_block_position function to get the (x, y) position of a block
def get_block_position(index, frame_width, frame_height, block_size=64):
    x = (index * block_size) % frame_width
    y = ((index * block_size) // frame_width) * block_size
    return x, y

# Define the find_mad function to calculate the Mean Absolute Difference
def find_mad(current_block, a_block):
    return np.sum(np.abs(np.subtract(current_block, a_block))) / (current_block.shape[0] * current_block.shape[1])


# Function to find the best matching block using motion compensation
def find_match(current_block, search_area):
    step = 32
    sa_height, sa_width = search_area.shape
    sa_centerY, sa_centerX = int(sa_height / 2), int(sa_width / 2)

    min_mad = float("+inf")
    minP = None

    while step >= 1:
        pointList = [
            (sa_centerX, sa_centerY), (sa_centerX + step, sa_centerY),
            (sa_centerX, sa_centerY + step), (sa_centerX + step, sa_centerY + step),
            (sa_centerX - step, sa_centerY), (sa_centerX, sa_centerY - step),
            (sa_centerX - step, sa_centerY - step), (sa_centerX + step, sa_centerY - step),
            (sa_centerX - step, sa_centerY + step)
        ]

        for p in range(len(pointList)):
            a_block = get_block_zone(pointList[p], search_area, current_block)
            mad = find_mad(current_block, a_block)
            if mad < min_mad:
                min_mad = mad
                minP = pointList[p]

        step = int(step / 2)

    px, py = minP
    px, py = px - int(current_block.shape[1] / 2), py - int(current_block.shape[0] / 2)
    px, py = max(0, px), max(0, py)

    match = search_area[py:py + current_block.shape[0], px:px + current_block.shape[1]]

    return match

# Define the find_match_hierarchical function to find the best matching block hierarchically
def find_match_hierarchical(target_block, search_area):
    best_match = None
    best_mad = float('inf')

    for y in range(0, search_area.shape[0] - target_block.shape[0] + 1):
        for x in range(0, search_area.shape[1] - target_block.shape[1] + 1):
            a_block = search_area[y:y + target_block.shape[0], x:x + target_block.shape[1]]
            mad = find_mad(target_block, a_block)
            if mad < best_mad:
                best_mad = mad
                best_match = a_block

    return best_match

# Function to create a predicted frame using hierarchical motion compensation
def create_predicted_hierarchical(predicted_frame, current_frame):
    predicted = predicted_frame.copy()
    current_blocks = split_frame(current_frame)

    for i in range(0, len(current_blocks)):
        current_block = current_blocks[i]
        iframe_block = find_match_hierarchical(current_block, predicted)
        
        x, y = get_block_position(i, current_frame.shape[1], current_frame.shape[0])
        predicted_block = predicted[y:y + current_block.shape[0], x:x + current_block.shape[1]]
        
        if predicted_block.shape == iframe_block.shape:
            predicted[y:y + current_block.shape[0], x:x + current_block.shape[1]] = iframe_block
        else:
            print(f"Shapes mismatch: predicted_block shape: {predicted_block.shape}, iframe_block shape: {iframe_block.shape}")

    return predicted

# Function to get the block zone for motion compensation
def get_block_zone(p, search_area, current_block):
    px, py = p
    px, py = px - 16, py - 16
    px, py = max(0, px), max(0, py)

    a_block = search_area[py:py + current_block.shape[0], px:px + current_block.shape[1]]

    try:
        assert a_block.shape == current_block.shape
    except:
        print("The blocks should have the same shape!")

    return a_block

# Function to find the center of a block
def find_center(x, y):
    return int(x + 8), int(y + 8)

# Function to find the residual frame
def find_residual(target_frame, predicted_frame):
    return np.subtract(target_frame, predicted_frame)

# Function to reconstruct the target frame
def reconstruct_target(residual_frame, predicted_frame):
    return np.add(residual_frame, predicted_frame)

#Main Program
frames = read_video_frames()
frames_debug = []
frames_debug = frames[:2]  # Take the first few frames from the frames list
                           # Testing with only a few frames for faster program runtime 
predicted_frames = predict_frames(frames_debug)

target_frames = []
residual_frames = []
reconstructed_frames = []

for i in range(len(frames_debug)):
    predicted_frame = create_predicted_hierarchical(predicted_frames[i], frames_debug[i])
    if i == 1:
        print("Predicted frame 1 created")
    residual_frame = find_residual(frames_debug[i], predicted_frame)
    if i == 1:
        print("Residual frame 1 created")
    reconstructed_frame = reconstruct_target(residual_frame, predicted_frame)

    target_frames.append(frames_debug[i])
    residual_frames.append(residual_frame)
    reconstructed_frames.append(reconstructed_frame)
    
    #Debuging
    if i == 1:
        print("Frame 1 motion compensated")
    if i == 2:
        print("Frame 2 motion compensated")
    if i == 5:
        print("Frame 55 motion compensated") 
    if i == 100:
        print("Frame 100 motion compensated")   
    if i == 135:
        print("135th frame motion compensated")    

# Visualize the error in prediciton frames
for residual_frame in residual_frames:
    # Display the error frame
    cv2.imshow('Residual Frame', residual_frame)
    
    # Add a small delay to display the frames at a reasonable speed, here it's 25 milliseconds
    cv2.waitKey(25)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
       
                 
# Visualize the reconstructed frames
for reconstructed_frame in reconstructed_frames:
    # Normalize pixel values to the valid range (0-255)
    reconstructed_frame_display = cv2.normalize(reconstructed_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the normalized reconstructed frame
    cv2.imshow('Reconstructed Frame', reconstructed_frame_display)
    
    # Add a small delay to display the frames at a reasonable speed, here it's 25 milliseconds
    cv2.waitKey(25)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break