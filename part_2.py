import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("image not found.")
    return img

def show_image_comparision(original, edited):
    plt.figure(figsize=(10, 2))
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
    action_history.append(f"Brightness {value}")
    return new_image

def adjust_contrast(image, value):
    new_image = cv2.convertScaleAbs(image, alpha=value)
    action_history.append(f"Contrast {value}")
    return new_image

def manual_blend(image1):
    print("Enter the path of the second image to blend with:")
    img_path = input()
    image2 = load_img(img_path)
    print("Enter alpha value for blending (0 to 1):")
    alpha = float(input())
    
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")
    
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    beta = 1 - alpha
    manual_img = alpha * image1 + beta * image2
    np.clip(manual_img, 0, 255)  
    manual_img = manual_img.astype(np.uint8)
    action_history.append(f"Blended with alpha {alpha}")

    return manual_img

def save_and_exit(image, filename):
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")
    view_history()
    exit()

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    action_history.append("Grayscaled")
    return gray_bgr

def apply_thresholding(image):
    print("""thresholding method: 
                1. Binary
                2. Inverse
          """)
    method = int(input("Select method (1 or 2): "))

    print("Enter the threshold value (0-255):")
    thresh_value = int(input())
    if not (0 <= thresh_value <= 255):
        raise ValueError("Threshold value must be between 0 and 255.")

    if method == 1:
        _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == 2:
        _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError("Invalid thresholding method. Use 'binary' or 'inverse'.")
    action_history.append(f"applied thresholding with value {thresh_value} {'binary' if method == 1 else 'inverse'}")
    return thresh_img

def add_padding(image):
    print("Enter the size of padding : ")
    size = int(input())
    if size <= 0:
        raise ValueError("Padding size must be a positive integer.")

    print("Choose padding type:")
    print("""Padding options:
                1. constant
                2. reflect
                3. replicate
                4. wrap
            """) 
    pad_option = int(input("Select padding option (1, 2, 3, or 4): "))
    if pad_option not in [1, 2, 3, 4]:
        raise ValueError("Invalid padding option. Choose 1, 2, 3, or 4.")
    
    if pad_option == 1:
        border_type = cv2.BORDER_CONSTANT
    elif pad_option == 2:
        border_type = cv2.BORDER_REFLECT
    elif pad_option == 3:
        border_type = cv2.BORDER_REPLICATE
    elif pad_option == 4:
        border_type = cv2.BORDER_WRAP
    
    print("""Padding ratios:
                1. 1:1 
                2. 16:9 
                3. 4:3 
                4. Custom Ratio
          """)
    pad_ratio = int(input("Select padding ratio (1, 2, 3 or 4): "))
    
    if pad_ratio not in [1, 2, 3, 4]:
        raise ValueError("Invalid padding ratio. Choose 1, 2, 3 or 4.")
    
    if pad_ratio == 1:
        target_ratio = 1
    elif pad_ratio == 2:
        target_ratio = 16 / 9
    elif pad_ratio == 3:
        target_ratio = 4 / 3
    elif pad_ratio == 4:
        ratio_input = input("Enter custom ratio (x:y): ")
        try:
            w,h = int(ratio_input.split(":"))
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
    action_history.append(f"added padding with size {size}, ratio {target_ratio} and type {pad_option}")

    return cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=[0, 0, 0])

def view_history():
    if not action_history:
        print("No actions performed yet.")
    else:
        print("Action History:")
        for i, action in enumerate(action_history, 1):
            print(f"{i}. {action}")

image_history = []
action_history = []

def main():
   
    while True:
        img_path = input("Enter the path of the image to edit: ")
        try:
            img = load_img(img_path)
            break
        except ValueError as e:
            print(e)
            continue
    edited_img = img.copy()
   
    while True:

        print("""==== Mini Photo Editor ====
              1. Adjust Brightness
              2. Adjust Contrast
              3. Convert to Grayscale
              4. Add Padding (choose border type)
              5. Apply Thresholding (binary or inverse)
              6. Blend with Another Image (manual alpha)
              7. Undo Last Operation
              8. View History of Operations
              9. Save and Exit
            """)
        option = input("Select an option: ")
        try:
            if option == "1":
                value = int(input("Enter brightness value(use -value for decreasing brgithness): "))
                edited_img = adjust_brightness(edited_img, value)
            elif option == "2":
                value = float(input("Enter contrast value : "))
                edited_img = adjust_contrast(edited_img, value)
            elif option == "3":
                edited_img = grayscale(edited_img)
            elif option == "4":
                edited_img = add_padding(edited_img)
            elif option == "5":
                edited_img = apply_thresholding(edited_img)
            elif option == "6":
                edited_img = manual_blend(edited_img)
            elif option == "8":
                view_history()
            elif option == "9":
                filename = input("Enter filename to : ")
                save_and_exit(edited_img, filename)

            else:
                print("Option not implemented yet.")
        
            show_image_comparision(img, edited_img)
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()