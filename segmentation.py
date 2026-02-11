import cv2
import numpy as np

def apply_borb_segmentation(img):
    # All these lines MUST be indented (pushed to the right)
    img_float = img.astype(np.float32)
    r, g, b = cv2.split(img_float)
    
    borb_feature = (2 * b) - r - g 
    
    borb_normalized = cv2.normalize(borb_feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(borb_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    
    # This return line MUST stay inside the function (indented)
    return borb_normalized, mask, segmented_img

# --- THIS IS THE "DRIVER CODE" WHERE YOU PUT THE IMAGE ---
# These lines are NOT indented (they start at the very edge)

# 1. Provide the path to your image file
image_path = 'dataset/valid/tomato_late/tomato_late_4.JPG' 

# 2. Read the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not find image at", image_path)
else:
    # 3. Call the function
    borb_view, leaf_mask, final_result = apply_borb_segmentation(image)

    # 4. Show the results
    cv2.imshow('Original', image)
    cv2.imshow('Segmented', final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()