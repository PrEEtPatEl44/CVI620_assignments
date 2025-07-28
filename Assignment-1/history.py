import cv2
import numpy as np
import matplotlib.pyplot as plt
action_history = []
image_history = []

def log_action(action):
    action_history.append(action)

def push_image(image):
    image_history.append(image.copy())

def undo_last_operation():
    if len(image_history) == 1:
        print("No actions to undo.")
    else:
        image_history.pop()
        action_history.pop()
     
    #action_history.append("Undo last operation")
    return image_history[-1]

# def get_last_image():
#     return image_history[-1]

def view_history():
    if not action_history:
        print("No actions performed yet.")
    else:
        for i, action in enumerate(action_history, 1):
            print(f"{i}. {action}")
