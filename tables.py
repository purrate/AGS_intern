import requests
import io
import cv2
import numpy as np
import pytesseract
import matplotlib.patches as patches

def detect_tables(image):
    """
    Sends the image to the Flask API for table detection and returns the bounding boxes.
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send request to the Flask API
        response = requests.post(
            "https://5a45-35-240-146-94.ngrok-free.app/predict",  # Replace with your ngrok URL
            files={"image": img_byte_arr}
        )
        response.raise_for_status()

        # Parse the response
        detections = response.json().get("detections", [])
        return detections
    except Exception as e:
        print(f"Error detecting tables: {e}")
        return []

def exclude_table_regions(content_blocks, table_boxes):
    """
    Excludes text blocks that overlap with table regions.
    """
    filtered_blocks = []
    for block in content_blocks:
        block_center_x = block['left'] + block['width'] / 2
        block_center_y = block['top'] + block['height'] / 2

        # Check if the block center is inside any table bounding box
        inside_table = False
        for table in table_boxes:
            x_min, y_min, x_max, y_max = table['bbox']
            if (x_min <= block_center_x <= x_max) and (y_min <= block_center_y <= y_max):
                inside_table = True
                break

        if not inside_table:
            filtered_blocks.append(block)

    return filtered_blocks
  
def visualize_tables(ax, table_boxes):
  """
  Draws rectangles around detected tables in the visualization.
  """
  for table in table_boxes:
      x_min, y_min, x_max, y_max = table['bbox']
      width = x_max - x_min
      height = y_max - y_min
      rect = patches.Rectangle(
          (x_min, y_min), width, height,
          linewidth=2, edgecolor='purple', facecolor='none', linestyle='--'
      )
      ax.add_patch(rect)
      ax.text(
          x_min, y_min - 10, f"Table ({table['confidence']:.2f})",
          color='purple', fontsize=10, backgroundcolor='white'
      )

def extract_table_text(img, table_boxes):
    """
    Extracts text content from detected tables.
    """
    table_data = []
    
    for i, table in enumerate(table_boxes):
        x_min, y_min, x_max, y_max = table['bbox']
        
        # Create a cropped image for the table region
        table_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Check if the table image is valid
        if table_img.size == 0:
            continue
            
        # Convert table image to grayscale
        if len(table_img.shape) > 2:
            table_gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        else:
            table_gray = table_img
            
        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(table_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Extract text using OCR
        custom_config = r'--oem 3 --psm 6'  # Assuming the table is structured text
        table_text = pytesseract.image_to_string(binary, config=custom_config)
        
        # Extract structured data with layout information
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Compile table text blocks
        text_blocks = []
        for j in range(len(data['text'])):
            if data['text'][j].strip() != '' and data['conf'][j] > 40:
                # Convert to global coordinates by adding table offset
                global_left = data['left'][j] + x_min
                global_top = data['top'][j] + y_min
                
                text_blocks.append({
                    'text': data['text'][j],
                    'left': global_left,
                    'top': global_top,
                    'width': data['width'][j],
                    'height': data['height'][j],
                    'conf': data['conf'][j]
                })
        
        # Determine if table has visible borders
        # A simple heuristic: Count edges in the image
        edges = cv2.Canny(table_gray, 50, 150)
        edge_count = np.sum(edges > 0)
        edge_density = edge_count / (table_gray.shape[0] * table_gray.shape[1])
        has_borders = edge_density > 0.1  # Threshold can be adjusted
        
        table_data.append({
            'id': i,
            'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],
            'confidence': float(table['confidence']),
            'text': table_text.strip(),
            'text_blocks': text_blocks,
            'has_borders': has_borders
        })
    
    return table_data