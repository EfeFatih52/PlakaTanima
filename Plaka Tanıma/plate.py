import cv2
import imutils
import numpy as np
import pytesseract
from tkinter import Tk, filedialog

# Tesseract OCR konumunu ayarla
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def open_photo_and_detect_plates():
    Tk().withdraw()  # Ana pencereyi gizle
    filename = filedialog.askopenfilename(title="Fotoğraf Seç", filetypes=(("Image Files", "*.jpg; *.jpeg; *.png; *.bmp"), ("All files", "*.*")))
    if filename:
        # Görüntüyü oku
        img = cv2.imread(filename)
        img = cv2.resize(img, (600, 400))

        # Plaka tanıma işlemini gerçekleştir
        plate_text, cropped_plate = detect_plates_in_image(img)
        print("Plaka Tanıma Programlaması\n")
        print("Plaka Numarası:", plate_text)

        # Görüntüyü ve kırpılmış plakayı göster
        cv2.imshow('Araba', img)
        cv2.imshow('Kirpildi', cropped_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_plates_in_image(image):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    # Kenarları algıla
    edged = cv2.Canny(gray, 30, 200) 

    # Konturları bul
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # Plaka konturunu bul
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # Plaka bulundu mu?
    if screenCnt is None:
        detected = False
        print("No contour detected")
    else:
        detected = True

    # Plaka bulunduysa
    if detected:
        # Plaka konturunu çiz
        cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

        # Plaka bölgesini maskele
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)

        # Plaka bölgesini kırp
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped_plate = gray[topx:bottomx + 1, topy:bottomy + 1]

        # Plaka metnini oku
        plate_text = pytesseract.image_to_string(cropped_plate, config='--psm 11')
        return plate_text, cropped_plate

# Fonksiyonu çağır
open_photo_and_detect_plates()
