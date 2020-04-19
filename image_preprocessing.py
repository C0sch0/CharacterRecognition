import cv2


def binarize(path):
    char_img = cv2.imread(path, 2)
    resized = cv2.resize(char_img, (250, 250))
    ret, thr_image = cv2.threshold(resized.copy(), 0, 255, cv2.THRESH_BINARY)
    # We don't use retVal for the moment, just take the thresholded image, so we ignore
    return thr_image


def segment_into_multiple(image_path, segmented_folder_path):
    preq_text = image_path[-6:-4]
    image = cv2.imread(image_path)
    original = image.copy()
    grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayscale, 0, 255, 0)
    all_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = all_contours[0] if len(all_contours) == 2 else all_contours[1]
    image_number = 0
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            segmented_img = original[y:y + h, x:x + w]
            cv2.imwrite("./{}/{}_{}.png".format(segmented_folder_path, preq_text, image_number), segmented_img)
            image_number += 1
