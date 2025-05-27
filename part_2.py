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
    return new_image

def adjust_contrast(image, value):
    new_image = cv2.convertScaleAbs(image, alpha=value)
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

    return manual_img

def save_and_exit(image, filename):
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")
    exit()

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

image_history = []

def main():
   
    while True:
        img_path = input("Enter the path of the image to edit: ")
        try:
            img = load_img(img_path)
            break
        except ValueError as e:
            print(e)
            continue

   
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
        if option == "1":
            value = int(input("Enter brightness value(use -value for decreasing brgithness): "))
            edited_img = adjust_brightness(edited_img, value)
        elif option == "2":
            value = float(input("Enter contrast value : "))
            edited_img = adjust_contrast(edited_img, value)
        elif option == "3":
            edited_img = grayscale(edited_img)
        elif option == "6":
            edited_img = manual_blend(edited_img)
        elif option == "9":
            filename = input("Enter filename to : ")
            save_and_exit(edited_img, filename)
        
        else:
            print("Option not implemented yet.")
        
        show_image_comparision(img, edited_img)


if __name__ == "__main__":
    main()