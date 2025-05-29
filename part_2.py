import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageEditor import *
from history import *

def save_and_exit(image):
    print("save image? (y/n)")
    save = input().lower()
    if save != 'y':
        print("Image not saved. Exiting without saving.")
        exit()
    else:
        filename = input("Enter the filename to save the image : ")
        cv2.imwrite(filename, image)
        print(f"Image saved as {filename}")
        view_history()
        exit()

def main():
   
    while True:
        img_path = input("Enter the path of the image to edit: ")
        try:
            img = load_img(img_path)
            #append the original image to history as the first entry
            image_history.append(img.copy())
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
                log_action(f"Brightness {value}")
            elif option == "2":
                value = float(input("Enter contrast value : "))
                edited_img = adjust_contrast(edited_img, value)
                log_action(f"Contrast {value}")
            elif option == "3":
                edited_img = grayscale(edited_img)
                log_action("Grayscaled")
            elif option == "4":

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
                border_types = {1: cv2.BORDER_CONSTANT, 2: cv2.BORDER_REFLECT, 3: cv2.BORDER_REPLICATE, 4: cv2.BORDER_WRAP} 
                pad_option = int(input("Select padding option (1, 2, 3, or 4): "))
                if pad_option not in [1, 2, 3, 4]:
                    raise ValueError("Invalid padding option. Choose 1, 2, 3, or 4.")
                border_type = border_types[pad_option]

                ratios = {1: 1, 2: 16/9, 3: 4/3, 4: -1}
                print("""Padding ratios:
1. 1:1 
2. 16:9 
3. 4:3 
4. Custom Ratio
                      """)
                pad_ratio = int(input("Select padding ratio (1, 2, 3 or 4): "))

                if pad_ratio not in [1, 2, 3, 4]:
                    raise ValueError("Invalid padding ratio. Choose 1, 2, 3 or 4.")

                target_ratio = ratios[pad_ratio]


                edited_img = add_padding(edited_img, target_ratio, border_type, size)
                log_action(f"Padding - size: {size}, ratio: {target_ratio}, type: {pad_option}")

            elif option == "5":
                
                print("""thresholding method: 
1. Binary
2. Inverse
                """)
                method = int(input("Select method (1 or 2): "))
                if method not in [1, 2]:
                    raise ValueError("Invalid method Choose 1 for Binary or 2 for Inverse.")
                
                print("Enter the threshold value (0-255):")
                thresh_value = int(input())
                if not (0 <= thresh_value <= 255):
                    raise ValueError("Threshold value must be between 0 and 255.")

                edited_img = apply_thresholding(edited_img, method, thresh_value)
                log_action(f"Thresholding - value: {thresh_value}, type: {'Binary' if method == 1 else 'Inverse'}")
                
            elif option == "6":

                print("Enter the path of the second image to blend with:")
                img_path = input()
                image2 = load_img(img_path)

                print("Enter alpha value for blending (0 to 1):")
                alpha = float(input())
                if not (0 <= alpha <= 1):
                    raise ValueError("Alpha must be between 0 and 1.")
                
                edited_img = manual_blend(edited_img, image2, alpha)
                log_action(f"Blending - img:{img_path},  alpha:{alpha}")

            elif option == "7":
                edited_img = undo_last_operation()
                continue
            elif option == "8":
                view_history()
                continue
            elif option == "9":
                save_and_exit(edited_img)
                continue
            else:
                print("Option not implemented yet.")

            
            push_image(edited_img.copy())
            show_image_comparision(img, edited_img)
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()