import torch
from PIL import Image
import pytesseract
import cv2
import numpy as np
import os

# roboflow
from roboflow import Roboflow

import random
import gtts
from playsound import playsound


def get_text(image):
    # preprocess the image by converting it to grayscale, blurring it, and computing an edge map
    gray = preprocess(image)

    # apply OCR to the image
    text = pytesseract.image_to_string(gray, lang="eng")

    # remove the temporary file
    os.remove(gray)

    return text


def preprocess(input_image: Image) -> str:
    # load the example image and convert it to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # opening
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # make a check to see if median blurring should be done to remove noise
    gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    return filename


def crop_image(new_x: int, new_y: int, new_w: int, new_h: int, original_image: Image) -> Image:
    # print(type(original_image))

    image_arr = np.array(original_image)

    # convert to int
    new_x = int(new_x)
    new_y = int(new_y)
    new_w = int(new_w)
    new_h = int(new_h)

    cropped_image = image_arr[new_y:new_y + new_h, new_x:new_x + new_w]

    final_image = Image.fromarray(cropped_image)

    return final_image


def read_text_from_image(image):
    # fix for tesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # use roboflow for text recognition
    '''rf = Roboflow(api_key="api key")
    project = rf.workspace().project("textfinder-h03qk")
    model = project.version(1).model'''

    model = torch.hub.load('ultralytics/yolov5', 'custom', path="yolo/best.pt")

    # print(dir(model.predict("images/test2.jpg", confidence=0, overlap=55)))

    # print(model.predict("images/test2.jpg", confidence=0, overlap=55).plot())

    #predictions = model.predict(image, confidence=0, overlap=55).predictions

    # for prediction in predictions:
    # print(prediction)

    # results = model.predict(image, confidence=0, overlap=55).json()
    results = model(image).crop()

    saved_results = []

    for result in results:
        #save image
        ran = random.randint(0, 1000)

        result.save(f"images/cropped-{ran}.jpg")

        # use cv2.imread to convert to cv2 image
        print("output2")
        print(get_text(cv2.imread(f"images/cropped-{ran}.jpg")))

        saved_results.append(get_text(cv2.imread(f"images/cropped-{ran}.jpg")))

    print(results)



    # crop each image then run tesseract on it
    '''for prediction in results['predictions']:
        cropped_image = crop_image(prediction['x'], prediction['y'], prediction['width'], prediction['height'],
                                   Image.open("images/test2.jpg"))

        ran_num = random.randint(0, 1000)

        cropped_image.save(f"images/cropped-{ran_num}.jpg")

        # use cv2.imread to convert to cv2 image
        print("output2")
        print(get_text(cv2.imread(f"images/cropped-{ran_num}.jpg")))

        saved_results.append(get_text(cv2.imread(f"images/cropped-{ran_num}.jpg")))

    for result in saved_results:
        ran_num = random.randint(0, 1000)

        tts = gtts.gTTS(result)
        tts.save(f"audio-{ran_num}.mp3")

        print("Playing audio...")
        playsound(f"audio-{ran_num}.mp3")'''

    return saved_results

    # crop the image and save it
    # crop_image(results['predictions'][0]['x'], results['predictions'][0]['y'], results['predictions'][0]['width'], results['predictions'][0]['height'],
    # Image.open("images/test2.jpg"))

    # load the image and convert it to grayscale
    '''image = cv2.imread("images/test2.jpg")

    print("output1")
    print(get_text(image))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # opening
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # make a check to see if median blurring should be done to remove noise
    gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can apply OCR to it
    filename = preprocess(image)

    # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print("output: ")
    print(text)

    # show the output images
    cv2.imshow("Image", image)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)'''
