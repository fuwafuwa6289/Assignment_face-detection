import cv2
import os
import time

# Open the default camera
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Load Haarcascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if Haarcascade is loaded successfully
if haar_cascade.empty():
    print("Error: Could not load Haarcascade XML file.")
    exit()

# Create directories for each member to save images
member_roles = [
    "P1_UXUI", "P2_CEO", "P3_Front-End Dev", "P4_Back-End Dev",
    "P5_Back-End Dev", "P6_Front-End Dev", "P7_Sale Engineer",
    "P8_System Analyst"
]

# Add nicknames for each member
member_nicknames = [
    "Fahfon", "Muay", "Nine", "J", 
    "First", "Mhooyong", "Golf", 
    "Boss"
]

base_output_dir = 'Output'

# Create folder for each member
for role in member_roles:
    os.makedirs(os.path.join(base_output_dir, role), exist_ok=True)

# Initialize counters for each member (10 images per member)
member_image_counts = {nickname: 0 for nickname in member_nicknames}

last_save_time = time.time()

# Start with the first member
current_member_index = 0
frame_count = 0  # Reset frame count for each member
countdown_time = 10  # Set initial countdown to 10 seconds
is_counting_down = False  # Flag to control countdown

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply face detection on the grayscale frame
    faces_rect = haar_cascade.detectMultiScale(gray_frame, 1.1, 9)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the frame if a face is detected and the member hasn't reached 10 images
        current_time = time.time()
        member_nickname = member_nicknames[current_member_index]
        member_folder = member_roles[current_member_index]

        if member_image_counts[member_nickname] < 10 and (current_time - last_save_time) >= 1:
            # Save the image for the current member
            frame_count += 1
            member_image_counts[member_nickname] += 1
            image_path = os.path.join(base_output_dir, member_folder, f'{frame_count}_{member_nickname}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            last_save_time = current_time

            # If the current member has reached 10 images, start countdown for the next member
            if member_image_counts[member_nickname] == 10:
                print(f"{member_nickname} has reached 10 images.")
                is_counting_down = True
                countdown_start_time = time.time()

        # Stop the process once all members have 10 images each
        if current_member_index == len(member_nicknames) - 1 and member_image_counts[member_nickname] == 10:
            # Only stop when the last member has reached 10 images
            print("All members have 10 images each. Exiting...")
            cam.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

    # Handle countdown logic for switching to the next member
    if is_counting_down:
        # Calculate remaining time for countdown
        elapsed_time = time.time() - countdown_start_time
        countdown_time = max(0, 10 - int(elapsed_time))

        # Display countdown timer on the frame
        if current_member_index + 1 < len(member_nicknames):
            countdown_text = f"Next: {member_nicknames[current_member_index + 1]} in {countdown_time} seconds"
        else:
            countdown_text = "All members finished"

        cv2.putText(frame, countdown_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # If countdown reaches 0, switch to the next member
        if countdown_time == 0 and current_member_index + 1 < len(member_nicknames):
            is_counting_down = False
            current_member_index += 1
            frame_count = 0  # Reset frame count for the next member

    # Write the frame to the output file
    out.write(frame)

    # Display the frame with face detection and countdown
    cv2.imshow('Camera - Face Detection', frame)

    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
