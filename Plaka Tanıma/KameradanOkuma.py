import cv2
import imutils
import numpy as np
import pytesseract

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_plate(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    # Detect edges
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Initialize variables
    screenCnt = None
    detected = 0

    # Iterate through contours
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            detected = 1
            break

    return screenCnt, detected

# Initialize the video capture object with camera index
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change to the appropriate index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Resize frame
    frame = cv2.resize(frame, (600, 400))

    # Detect plate
    screenCnt, detected = detect_plate(frame)

    if detected:
        # Draw contours
        cv2.drawContours(frame, [screenCnt], -1, (0, 0, 255), 3)

        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [screenCnt], -1, (255, 255, 255), -1)

        # Apply mask
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Crop plate
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped_plate = masked[topx:bottomx + 1, topy:bottomy + 1]

        # Perform OCR
        plate_text = pytesseract.image_to_string(cropped_plate, config='--psm 11')
        print("Plaka Tanıma Programlaması\n")
        print("Plaka Numarası:", plate_text)

    # Resize frame and cropped plate for display
    frame = cv2.resize(frame, (500, 300))
    if detected:
        cropped_plate = cv2.resize(cropped_plate, (400, 200))

    # Display frames
    cv2.imshow('Araba', frame)
    if detected:
        cv2.imshow('Kirpildi', cropped_plate)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
