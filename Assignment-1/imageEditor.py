import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("image not found.")
    return img

def show_image_comparision(original, edited):    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)

    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    plt.subplot(122)
    
    plt.imshow(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB))
    plt.title("Edited")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def adjust_brightness(image, value):
    new_image = cv2.convertScaleAbs(image, beta=value)
    # action_history.append(f"Brightness {value}")
    return new_image

def adjust_contrast(image, value):
    new_image = cv2.convertScaleAbs(image, alpha=value)
    # action_history.append(f"Contrast {value}")
    return new_image

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # action_history.append("Grayscaled")
    return gray_bgr

def apply_thresholding(image, method, thresh_value):
    

    if method == 1:
        _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == 2:
        _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError("Invalid thresholding method. Use 'binary' or 'inverse'.")
    # action_history.append(f"Thresholding - value: {thresh_value}, type: {'Binary' if method == 1 else 'Inverse'}")
    return thresh_img

def add_padding(image, target_ratio, border_type, size):
    if target_ratio == -1:
        ratio_input = input("Enter custom ratio (x:y): ")
        try:
            w_str,h_str = ratio_input.split(":")
            w = int(w_str)
            h = int(h_str)
        except ValueError:
            raise ValueError("Invalid ratio format. Use 'x:y' format.")
        target_ratio = w / h
    
    current_ratio = image.shape[1] / image.shape[0]

    if current_ratio > target_ratio:
        #pad the height, width remains same
        new_h = int(image.shape[1] / target_ratio)
        pad_h = new_h - image.shape[0]
        top = pad_h // 2
        bottom = pad_h - top
        left = right = 0
    else:
        #pad the width, height remains same
        new_w = int(image.shape[0] * target_ratio)
        pad_w = new_w - image.shape[1]
        left = pad_w // 2
        right = pad_w - left
        top = bottom = 0
    # action_history.append(f"Padding - sizee: {size}, ratio: {target_ratio} and type: {pad_option}")

    return cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=[0, 0, 0])

def manual_blend(image1, image2, alpha):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    beta = 1 - alpha
    manual_img = alpha * image1 + beta * image2
    np.clip(manual_img, 0, 255)  
    manual_img = manual_img.astype(np.uint8)
    # action_history.append(f"Blending - img:{img_path},  alpha:{alpha}")

    return manual_img