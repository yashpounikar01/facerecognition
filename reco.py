import cv2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load CSV file for attendance
attendance_file = 'attendance.csv'

# Load or create attendance dataframe
try:
    attendance_df = pd.read_csv(attendance_file)
except FileNotFoundError:
    # If file doesn't exist, create a new DataFrame
    attendance_df = pd.DataFrame(columns=['Name', 'Time'])

# Initialize video capture
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Recognize face (This part needs to be implemented based on your recognition method)

        # Assuming recognition part returns the name of the person
        name = "John Doe"  # Example name, replace with actual name

        # Get current time
        time_now = datetime.now().strftime("%H:%M:%S")

        # Check if name already marked for today
        if name not in attendance_df['Name'].values:
            # Append name and time to DataFrame
            attendance_df = attendance_df.append({'Name': name, 'Time': time_now}, ignore_index=True)
            # Write DataFrame to CSV file
            attendance_df.to_csv(attendance_file, index=False)

    # Display the resulting frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Face Recognition')
    plt.show()

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
