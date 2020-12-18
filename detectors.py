"""
Stores detectors and auxiliary functions
"""

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_local

import string
import pytesseract


DOC_COLOR_LOWER = (127, 127, 127)
DOC_COLOR_UPPER = (240, 240, 240)


def doc_threshold(image, white=True):
    """Proper thresholding for document"""
    if white:
        mask = cv2.inRange(image, DOC_COLOR_LOWER, DOC_COLOR_UPPER)
        image = cv2.bitwise_and(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


def order_points(pts):
    """Order points to get proper rectangle form"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Warping document from image to normal rectangle"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width1), int(width2))
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height1), int(height2))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))
    return warped


def detect_document(img_path, pdf_path, binarize=True, white=True):
    """Document detector: inputs image and outputs document in pdf"""
    if pdf_path.count('.pdf') != 1:
        raise Exception("Failed: Use proper name with .pdf extension for your document or choose default one")
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    thresh = doc_threshold(image.copy(), white=white)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:7]
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
    if screen_cnt is None:
        raise Exception("Failed: Not able to detect edges")

    warped = four_point_transform(gray, screen_cnt.reshape(4, 2))
    if warped.shape[0] < warped.shape[1]:
        warped = warped.T
    img_warped = Image.fromarray(warped)
    img_warped.save(pdf_path)

    thresh_local = threshold_local(warped, 7, offset=7, method="gaussian")
    binarized = (warped > thresh_local).astype("uint8") * 255
    # white border
    border_x, border_y = (0.02 * np.array(binarized.shape)).astype(int)
    border = np.ones((binarized.shape[0] - 2 * border_x, binarized.shape[1] - 2 * border_y))
    border = np.pad(border, ((border_x, border_x), (border_y, border_y)))
    binarized = np.uint8(binarized + 255 * (1 - border))
    if binarize:
        pdf_path = pdf_path.rsplit('.', 1)
        pdf_path = pdf_path[0] + '_bin.' + pdf_path[1]
        img_binarized = Image.fromarray(binarized)
        img_binarized.save(pdf_path)
    return warped, binarized


def ED(s, t):
    """Edit/Levenshtein distance between words"""
    s = ' ' + s
    t = ' ' + t
    D = [[0 for j in range(len(s))] for i in range(len(t))]
    for i in range(len(s)):
        D[0][i] = i
    for i in range(len(t)):
        D[i][0] = i
    for i in range(1, len(t)):
        for j in range(1, len(s)):
            if t[i] == s[j]:
                D[i][j] = min(D[i - 1][j - 1], D[i - 1][j] + 1, D[i][j - 1] + 1)
            else:
                D[i][j] = min(D[i - 1][j - 1] + 1, D[i - 1][j] + 1, D[i][j - 1] + 1)
    return D[-1][-1]


def ind_of_sign(words, target_words):
    """Derive indices of words in document that are (close to) target words"""
    target_lengths = [len(tw) for tw in target_words]
    min_target_dist = sum(target_lengths) // (2 * len(target_words))
    min_target_len, max_target_len = min(target_lengths), max(target_lengths)
    indices = []
    for index, s in enumerate(words):
        s = str(s).lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        dist = min([ED(i, s) for i in target_words])
        if dist < min_target_dist and min_target_len <= len(s) <= max_target_len:
            indices.append(index)
    return indices


def get_box_from(document, target_words, lang):
    """Recognize words in document and return bounding boxes for them"""
    df = pytesseract.image_to_data(document, lang=lang, output_type=pytesseract.Output.DATAFRAME)
    df = df[df['conf'] != -1]
    df = df[~df['text'].isna()]
    indices = ind_of_sign(df['text'], target_words)
    if len(indices) == 0:
        raise Exception("None of target words is found in text")
    boxes = []
    for index in indices:
        row = df.iloc[index]
        boxes.append(((row['left']-10, row['top']-10), (row['left'] + row['width']+10, row['top']+row['height']+10)))
    return boxes
