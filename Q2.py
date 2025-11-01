import cv2
import numpy as np 


# Function to predict frames and create a predicted frame list
def predict_frames(frames):
    predicted_frames = [frames[0]]  # Initial frame (i-frame) is the same as the first actual frame

    for i in range(1, len(frames)):
        predicted_frames.append(frames[i - 1])

    return predicted_frames

# Function to calculate Mean Absolute Difference
def find_mad(current_block, a_block):
    return np.sum(np.abs(np.subtract(current_block, a_block))) / (current_block.shape[0] * current_block.shape[1])

# Function to find the best matching block using motion compensation
def find_match(current_block, search_area):
    step = 4
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
    px, py = px - 8, py - 8
    px, py = max(0, px), max(0, py)

    match = search_area[py:py + 16, px:px + 16]

    return match

# Function to create a predicted frame using motion compensation
def create_predicted(frame, target_frame):
    height, width = frame.shape
    vertical_mblocks, horizontal_mblocks = divide_frame(frame)

    predicted = np.ones((height, width)) * 255
    bcount = 0

    for y in range(0, int(vertical_mblocks * 16), 16):
        for x in range(0, int(horizontal_mblocks * 16), 16):
            bcount += 1
            target_block = target_frame[y:y + 16, x:x + 16]
            search_area = find_search_area(x, y, frame)
            iframe_block = find_match(target_block, search_area)

            predicted[y:y + 16, x:x + 16] = iframe_block

    assert bcount == int(horizontal_mblocks * vertical_mblocks)

    return predicted

# Function to divide a frame into macroblocks and get their coordinates
def divide_frame(frame):
    height, width = frame.shape
    vertical_mblocks = int(height / 16)
    horizontal_mblocks = int(width / 16)

    return vertical_mblocks, horizontal_mblocks

# Function to find the search area for motion compensation
def find_search_area(x, y, frame):
    height, width = frame.shape
    center_x, center_y = find_center(x, y)

    search_x = max(0, center_x - 24)
    search_y = max(0, center_y - 24)

    search_area = frame[search_y:min(search_y + 48, height), search_x:min(search_x + 48, width)]

    return search_area

# Function to get the block zone for motion compensation
def get_block_zone(p, search_area, current_block):
    px, py = p
    px, py = px - 8, py - 8
    px, py = max(0, px), max(0, py)

    a_block = search_area[py:py + 16, px:px + 16]

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

#Main Program

# Read the video
video = 'ball.mp4'
cap = cv2.VideoCapture(video)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error opening video file")

# Get the frame rate of the original video
fps = int(cap.get(cv2.CAP_PROP_FPS))

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

# Now we have a list of grayscale frames for further processing

predicted_frames = predict_frames(frames) #predicted_frames[i] now has frames[i-1](previous frame)except for the i-frame
background = frames[0]
target_frames = []
residual_frames = []
removed_object_frames = [predicted_frames[0]]   

for i in range(1, len(frames)):
    # Create the predicted frame using the previous removed object frame
    predicted_frame = create_predicted(removed_object_frames[i-1], frames[i]) 
    
    # Calculate the residual frame
    residual_frame = find_residual(frames[i], predicted_frame)
    
    # Threshold the residual frame to create a binary mask
    threshold = 30  # Adjust as needed
    _, mask = cv2.threshold(residual_frame, threshold, 255, cv2.THRESH_BINARY)
    
    # Replace moving object regions in the current frame with background regions
    frame_with_object_removed = frames[i].copy()
    frame_with_object_removed[mask == 255] = background[mask == 255]
   
    target_frames.append(frames[i])
    residual_frames.append(residual_frame)
    removed_object_frames.append(frame_with_object_removed)

# Visualize the error in prediciton frames
for residual_frame in residual_frames:
    # Display the error frame
    cv2.imshow('Residual Frame', residual_frame)
    
    # Add a small delay to display the frames at a reasonable speed, here it's 25 milliseconds
    cv2.waitKey(25)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Close all OpenCV windows
cv2.destroyAllWindows()
 
#Create a new video without the object

#Get frame shape
frame_width = frames[0].shape[1]
frame_height = frames[0].shape[0]

# Define the codec and create a VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'H264')
final_video = cv2.VideoWriter('removed_ball.mp4', fourcc, fps, (frame_width, frame_height))


# Visualize the reconstructed frames and write them into final_video
for frame_with_object_removed in removed_object_frames:
    # # Normalize pixel values to the valid range (0-255)
    # reconstructed_frame_display = cv2.normalize(reconstructed_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Resize the frame_with_object_removed to match the original frame size
    frame_with_object_removed = cv2.resize(frame_with_object_removed, (frame_width, frame_height)) 
    
    final_video.write(frame_with_object_removed)
 
    # Display each frame
    cv2.imshow('Object Removal', frame_with_object_removed)
    
    # Add a small delay to display the frames at a reasonable speed, here it's 25 milliseconds
    cv2.waitKey(25)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the VideoWriter object
final_video.release()    
           
