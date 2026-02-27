"""
Project: Automated Optical Inspection (AOI) - Feature Extractor
Author: Sudarshan (Mechatronics Engineering, NHIT)
Description: Vision system for PCB component analysis using Adaptive Thresholding,
             Morphological Noise Suppression, and ROI isolation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class ScientificFeatureExtractor:
    def __init__(self, pixel_to_unit_ratio=0.05, unit_name="mm"):
        self.ratio = pixel_to_unit_ratio
        self.unit = unit_name
        self.results = []

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # ROI Selection: Focusing on the PCB and cropping out the side rail/background
        # Coordinates based on your 3000x4500 sample image
        roi_img = img[300:2800, 1400:3900]

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)
        return roi_img, gray, blurred

    def segment_features(self, blurred_img):
        # Adaptive thresholding for uneven lighting on the PCB surface
        thresh = cv2.adaptiveThreshold(
            blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological Opening: Effectively "vacuums" up small white background noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        return cleaned_thresh

    def analyze_contours(self, roi_img, thresh_img):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated_img = roi_img.copy()
        self.results = []

        for i, cnt in enumerate(contours):
            area_px = cv2.contourArea(cnt)

            # Area Filter: Ignoring artifacts smaller than 800px to keep only major components
            if area_px < 800:
                continue

            area_physical = area_px * (self.ratio ** 2)
            M = cv2.moments(cnt)
            cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

            self.results.append({
                "Feature_ID": len(self.results) + 1,
                f"Area_{self.unit}2": round(area_physical, 4),
                "Centroid": (cX, cY)
            })

            # Labeling the identified features
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(annotated_img, f"ID:{len(self.results)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_img

def main():
    image_path = "sample.jpg"
    extractor = ScientificFeatureExtractor(pixel_to_unit_ratio=0.05, unit_name="mm")

    try:
        roi, gray, blurred = extractor.preprocess_image(image_path)
        thresh = extractor.segment_features(blurred)
        final_visual = extractor.analyze_contours(roi, thresh)

        # Output results to CSV
        df = pd.DataFrame(extractor.results)
        df.to_csv("pcb_analysis.csv", index=False)

        # Matplotlib visualization (Display-safe for Arch/Wayland)
        plt.figure(figsize=(15, 10))
        plt.rcParams['figure.dpi'] = 150

        titles = ["1. ROI Crop", "2. Gaussian Blur", "3. Morphological Cleaning", "4. Feature Extraction"]
        images = [roi, blurred, thresh, final_visual]

        for i in range(4):
            plt.subplot(2, 2, i+1); plt.title(titles[i])
            if i == 0 or i == 3:
                plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(images[i], cmap='gray')

        plt.tight_layout()
        print(f"Success! {len(extractor.results)} components identified. Data saved to 'pcb_analysis.csv'.")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
