import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        print("Loaded data:")
        print(np_path_XYs)
        
        path_XYs = []
        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []
            for j in np.unique(npXYs[:, 0]):
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY.astype(np.int32))
            path_XYs.append(XYs)
        
        return path_XYs
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def classify_shape(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 2:
        return "Straight line"
    elif num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            return "Rounded rectangle"
        else:
            return "Rectangle"
    elif num_vertices > 6:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.85:
            return "Circle"
        elif 0.5 < circularity < 0.85:
            return "Ellipse"
        else:
            return "Star shape"
    elif 5 <= num_vertices <= 6:
        return "Regular Polygon"
    
    return "Unknown Shape"

def plot_input_data(path_XYs, input_image_path):
    plt.figure(figsize=(5, 5))
    for path_contours in path_XYs:
        for contour in path_contours:
            plt.plot(contour[:, 0], contour[:, 1], 'bo-') 
    plt.gca().invert_yaxis() 
    plt.title("Input CSV Data")
    plt.axis('equal')
    plt.savefig(input_image_path) 
    plt.show()

def detect_shapes_refined_from_csv(csv_path):
    path_XYs = read_csv(csv_path)
    if not path_XYs:
        return

    filename = os.path.splitext(os.path.basename(csv_path))[0]

    input_image_path = f"{filename}_input.png"
    output_image_path = f"{filename}_output.png"

    plot_input_data(path_XYs, input_image_path)
    
    output = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    for path_contours in path_XYs:
        for contour in path_contours:
            shape = classify_shape(contour)
            print(f"Detected shape: {shape}")
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)


            if area < 100:
                continue

            if len(approx) == 3:
                cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # Triangle
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Square
                else:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rectangle
            elif len(approx) == 5:
                cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # Pentagon
            elif len(approx) > 6:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.7 < circularity < 1.3:
                    cv2.circle(output, center, radius, (0, 0, 255), 2)  # Circle
                else:
                    cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # Other shapes

    # Save & display output image
    cv2.imwrite(output_image_path, output)
    print(f"Input image saved at {input_image_path}")
    print(f"Output image saved at {output_image_path}")
    output_image = Image.open(output_image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(output_image)
    plt.title("Detected Shapes")
    plt.axis('off')
    plt.show()

def main():
    csv_path = r"E:\PROJ 2\CURVETOPIA-main\problems\isolated.csv"
    
    if os.path.exists(csv_path):
        print(f"Processing CSV file at {csv_path}.")
        detect_shapes_refined_from_csv(csv_path)
    else:
        print(f"The CSV file does not exist at {csv_path}.")

if __name__ == "__main__":
    main()
