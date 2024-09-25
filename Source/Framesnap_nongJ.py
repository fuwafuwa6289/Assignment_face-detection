import numpy as np
import json
import cv2
import os

def create_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters:
        directory (str): The path of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    """
    Get the first available identifier.

    Parameters:
        directory (str): The path of the directory of images.
    """
    user_ids = []
    for filename in os.listdir(directory):
        # Check if the filename matches the expected format
        parts = os.path.split(filename)[-1].split("-")
        if len(parts) > 1 and parts[1].isdigit():
            number = int(parts[1])
            user_ids.append(number)
    
    user_ids = sorted(list(set(user_ids)))
    max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
    for i in range(1, max_user_ids + 1):
        if i not in user_ids:
            return i
    return max_user_ids

def save_name(face_id: int, face_name: str, filename: str) -> None:
    """
    Save the face ID and name pair in the JSON file without overwriting.
    
    Parameters:
        face_id (int): The identifier for the face.
        face_name (str): The name associated with the face_id.
        filename (str): The JSON filename.
    """
    # Check if the file exists and load its contents
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                names_json = json.load(file)
            except json.JSONDecodeError:
                names_json = {}
    else:
        names_json = {}

    # Add the new face_id and name to the dictionary
    names_json[str(face_id)] = face_name

    # Write the updated dictionary back to the JSON file
    with open(filename, 'w') as file:
        json.dump(names_json, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Get the current directory where the Python script is located
    current_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Define paths relative to the script location
    base_directory = os.path.join(current_directory, 'images')
    cascade_classifier_filename = './haarcascade_frontalface_default.xml'
    names_json_filename = os.path.join(current_directory, 'names.json')

    # Create 'images' directory if it doesn't exist
    create_directory(base_directory)
    
    # Load the pre-trained face cascade classifier
    faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)
    
    # Open a connection to the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    
    # Set camera dimensions
    cam.set(3, 640)
    cam.set(4, 480)
    
    # Initialize face capture variables
    count = 0
    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(base_directory)
    save_name(face_id, face_name, names_json_filename)
    
    # Create a directory for the specific user
    user_directory = os.path.join(base_directory, face_name)
    create_directory(user_directory)
    
    print(f'\n[INFO] Initializing face capture for {face_name}. Look at the camera and wait...')

    while True:
        # Read a frame from the camera
        ret, img = cam.read()
    
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
            # Increment the count for naming the saved images
            count += 1

            # Save the captured image into the user's specific directory
            cv2.imwrite(f'{user_directory}/Users-{face_id}-{count}.jpg', gray[y:y+h, x:x+w])
    
            # Display the image with rectangles around faces
            cv2.imshow('image', img)
    
        # Press Escape to end the program
        k = cv2.waitKey(100) & 0xff
        if k == 27:  # Press 'ESC' to break the loop
            break
    
        # Take 150 face samples and stop video
        elif count >= 150:
            # Reset count and create new directory for the next capture
            count = 0
            face_name = input('\nEnter user name for next session and press <return> -->  ')
            face_id = get_face_id(base_directory)
            save_name(face_id, face_name, names_json_filename)
            
            # Create a new directory for the next user
            user_directory = os.path.join(base_directory, face_name)
            create_directory(user_directory)

    print('\n[INFO] Success! Exiting Program.')
    
    # Release the camera
    cam.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
