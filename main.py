import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import json
from datetime import datetime

from tables import detect_tables, exclude_table_regions, extract_table_text
from visualisation import create_enhanced_visualization
from header_footer import extract_header_footer_text
from layout import calculate_layout_features, classify_layout_with_features
from seperator import find_optimal_separator, analyze_vertical_whitespace

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def detect_columns_with_density(x_midpoints, width):
    """
    Uses density-based peak detection to identify columns.
    """
    # Generate density histogram with more bins for precision
    hist, bin_edges = np.histogram(x_midpoints, bins=50, range=(0, width))
    
    # Apply Gaussian smoothing to reduce noise
    smoothed_hist = gaussian_filter1d(hist, sigma=1.5)
    
    # Find peaks with minimum separation of 20% of page width and prominence
    min_distance = int(width * 0.2 / (width / 50))  # Convert to bin count
    peaks, _ = find_peaks(smoothed_hist, distance=min_distance, prominence=max(smoothed_hist) * 0.2) if len(smoothed_hist) > 0 else ([], None)
    
    # Convert peak indices back to x-coordinates
    peak_positions = [bin_edges[p] + (bin_edges[p+1] - bin_edges[p])/2 for p in peaks]
    
    # Determine layout based on peaks
    if len(peak_positions) >= 2:
        # Additional verification - check if peaks are well separated
        separations = np.diff(sorted(peak_positions))
        if np.max(separations) > width * 0.15:  # Adaptive threshold
            return "Dual-column", peak_positions
    
    return "Single-column", peak_positions

def cluster_columns_with_dbscan(x_midpoints, width):
    """
    Uses DBSCAN for more robust column detection.
    """
    if len(x_midpoints) < 5:
        return "Single-column", [np.mean(x_midpoints)] if x_midpoints else []
    
    # Reshape for DBSCAN
    X = np.array(x_midpoints).reshape(-1, 1)
    
    # Adaptive epsilon based on page width
    eps = width * 0.1  # 10% of page width
    
    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=3).fit(X)
    labels = db.labels_
    
    # Count samples in each cluster (excluding noise at -1)
    unique_labels = set(labels) - {-1}
    cluster_counts = {label: np.sum(labels == label) for label in unique_labels}
    
    # Get cluster centers
    centers = []
    for label in unique_labels:
        if cluster_counts[label] > len(x_midpoints) * 0.1:  # Filter small clusters
            centers.append(np.mean(X[labels == label]))
    
    centers = sorted(centers)
    
    # Determine layout type
    if len(centers) >= 2 and max(np.diff(centers)) > width * 0.15:
        return "Dual-column", centers
    else:
        return "Single-column", centers

def verify_column_alignment(column_blocks, width):
    """
    Verifies the vertical alignment within a column by measuring variance.
    Lower variance indicates better column alignment.
    """
    if not column_blocks:
        return 0
        
    # Extract left edges and widths
    left_edges = [block['left'] for block in column_blocks]
    right_edges = [block['left'] + block['width'] for block in column_blocks]
    
    # Calculate variance of edges (lower is better aligned)
    left_variance = np.var(left_edges)
    right_variance = np.var(right_edges)
    
    # Calculate alignment score (inverse of variance, normalized)
    max_var = max(left_variance, right_variance)
    if max_var == 0:
        return 1.0  # Perfect alignment
    
    alignment_score = 1.0 - (max_var / (width * width * 0.1))  # Normalize
    return max(0, min(1, alignment_score))  # Clamp between 0 and 1

def analyze_column_separation(columns, width, content_blocks):
    """
    Analyzes the "emptiness" of space between columns.
    """
    if len(columns) < 2:
        return 0
    
    # Sort columns by x position
    col_centers = []
    for col in columns:
        if col:
            x_center = np.mean([b['left'] + b['width']/2 for b in col])
            col_centers.append(x_center)
    
    col_centers = sorted(col_centers)
    
    if len(col_centers) < 2:
        return 0
    
    # Find the midpoint between columns
    col1_rightmost = max([b['left'] + b['width'] for b in columns[0]]) if columns[0] else 0
    col2_leftmost = min([b['left'] for b in columns[1]]) if columns[1] else width
    
    # Calculate separation
    separation = col2_leftmost - col1_rightmost
    normalized_separation = separation / width
    
    # Count blocks in the separation area
    blocks_in_gap = 0
    for block in content_blocks:
        block_center = block['left'] + block['width']/2
        if col1_rightmost < block_center < col2_leftmost:
            blocks_in_gap += 1
    
    # Calculate separation score
    if blocks_in_gap > 0 and len(content_blocks) > 0:
        normalized_separation *= (1 - (blocks_in_gap / len(content_blocks)))
    
    return normalized_separation

def determine_layout(content_blocks, width):
    """
    Determines layout using multiple methods and verification steps.
    """
    if len(content_blocks) < 10:
        return "Single-column", []
    
    # Extract x midpoints
    x_midpoints = [block['left'] + block['width']/2 for block in content_blocks]
    
    # Method 1: Density-based peak detection
    layout1, centers1 = detect_columns_with_density(x_midpoints, width)
    
    # Method 2: DBSCAN clustering
    layout2, centers2 = cluster_columns_with_dbscan(x_midpoints, width)
    
    # If both methods agree on dual-column, proceed with verification
    if layout1 == "Dual-column" and layout2 == "Dual-column":
        # Use centers from DBSCAN for grouping
        col1_blocks = []
        col2_blocks = []
        
        # Midpoint between columns for classification
        mid_separator = (centers2[0] + centers2[1]) / 2
        
        for block in content_blocks:
            block_center = block['left'] + block['width']/2
            if block_center < mid_separator:
                col1_blocks.append(block)
            else:
                col2_blocks.append(block)
        
        # Verify column alignment
        col1_alignment = verify_column_alignment(col1_blocks, width)
        col2_alignment = verify_column_alignment(col2_blocks, width)
        
        # Verify separation
        separation_score = analyze_column_separation([col1_blocks, col2_blocks], width, content_blocks)
        
        # Make final decision based on all evidence
        if col1_alignment > 0.7 and col2_alignment > 0.7 and separation_score > 0.1:
            return "Dual-column", centers2
    
    # If methods disagree or verification fails, use secondary evidence
    if layout1 == "Dual-column" or layout2 == "Dual-column":
        # Some evidence for dual columns, but not conclusive
        return "Possible dual-column", centers1 if layout1 == "Dual-column" else centers2
    
    return "Single-column", centers1



def combined_pdf_layout_analysis(pdf_path, output_dir=None, save_json=True):
    """
    Combined algorithm that leverages advanced column detection with JSON output:
    1. Uses density-based peak detection
    2. Applies DBSCAN clustering
    3. Verifies column consistency
    4. Enhanced visualization
    5. Stores results in JSON format
    
    Parameters:
    - pdf_path: Path to the PDF file
    - output_dir: Optional custom output directory
    - save_json: Whether to save the JSON output to a file
    """
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path), "layout_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    
    print(f"\nAnalyzing layout for: {pdf_name}")
    print("-" * 70)
    
    # Initialize JSON structure
    json_result = {
        "pdf_name": pdf_name,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_pages": len(images),
        "pages": []
    }
    
    # Process each page
    page_results = []
    
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"\nProcessing page {page_num} of {len(images)}...")
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # For matplotlib
        
        # Detect tables
        table_boxes = detect_tables(img)
        print(f"Detected {len(table_boxes)} tables on page {page_num}")
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # Grayscale for better OCR
        height, width = img_cv.shape[:2]
        
        # IMPROVEMENT: Preprocess image for better OCR
        # Apply adaptive thresholding to improve text detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Run OCR with custom config (improved detection)
        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Extract x-coordinates of text blocks with higher confidence threshold
        x_coords = [data['left'][j] for j in range(len(data['text'])) 
                   if data['text'][j].strip() != '' and data['conf'][j] > 40]  # Increased confidence threshold
        
        # Analyze distribution of x-coordinates
        if not x_coords:
            result = "No text detected"
            page_data = {
                "page_number": page_num,
                "layout": result,
                "header": {"text": "", "blocks": []},
                "footer": {"text": "", "blocks": []},
                "tables": [],
                "columns": []
            }
            json_result["pages"].append(page_data)
            page_results.append({"page": page_num, "layout": result})
            print(f"  Page {page_num}: {result}")
            continue
        
        # STAGE 2: Extract text blocks for detailed analysis with higher confidence
        # --------------------------------------------------
        text_blocks = []
        for j in range(len(data['text'])):
            if data['text'][j].strip() != '' and data['conf'][j] > 40:  # Higher confidence threshold
                text_blocks.append({
                    'text': data['text'][j],
                    'left': data['left'][j],
                    'top': data['top'][j],
                    'width': data['width'][j],
                    'height': data['height'][j],
                    'conf': data['conf'][j]
                })
        content_blocks = exclude_table_regions(text_blocks, table_boxes)
        
        # STAGE 3: Header/Footer detection
        # --------------------------------
        # Simple header/footer estimate (10% of page)
        header_height = int(height * 0.1)
        footer_start = height - int(height * 0.1)
        
        # Separate text blocks by region
        header_blocks = [block for block in text_blocks if block['top'] < header_height]
        footer_blocks = [block for block in text_blocks if block['top'] > footer_start]
        content_blocks = [block for block in text_blocks 
                         if block['top'] >= header_height and block['top'] <= footer_start]
        
        # Extract header and footer text with coordinates
        header_data, footer_data = extract_header_footer_text(header_blocks, footer_blocks)
        
        # STAGE 4: Extract table data with text content
        # --------------------------------------------
        table_data = extract_table_text(img_cv, table_boxes)
        
        # STAGE 5: Enhanced column detection
        
        layout_type, centers = determine_layout(content_blocks, width)
                # Analyze vertical whitespace for column separation
        whitespace_profile, whitespace_separators = analyze_vertical_whitespace(gray, width, height)
        print(f"  Detected {len(whitespace_separators)} potential column separators")
        
        # Calculate layout features for more robust classification
        layout_features = calculate_layout_features(
            content_blocks, header_blocks, footer_blocks, table_boxes, 
            width, height, whitespace_profile
        )

        # Get layout classification and confidence based on features
        feature_layout, feature_confidence = classify_layout_with_features(layout_features)
        print(f"  Feature-based classification: {feature_layout} (confidence: {feature_confidence:.2f})")

        # Combine feature-based classification with existing methods
        if feature_confidence > 0.8:
            # High confidence in feature-based classification
            layout_type = feature_layout
        elif layout_type == feature_layout:
            # Both methods agree
            layout_type = feature_layout
        elif feature_layout == "Dual-column" and layout_type == "Possible dual-column":
            # Feature-based supports existing suspicion
            layout_type = "Dual-column"
        elif feature_layout == "Possible dual-column" and layout_type == "Single-column":
            # Feature-based suggests possibility not caught by other methods
            layout_type = "Possible dual-column"
        
        # Process column data for JSON
        column_data = []
        # Incorporate whitespace information into column detection
        if layout_type == "Dual-column" or "Possible dual-column" and len(centers) >= 2:
            # Find the whitespace separator closest to the midpoint between column centers
            mid_point = (centers[0] + centers[1]) / 2
            
            if whitespace_separators:
                # Find closest whitespace separator to the midpoint
                optimal_separator = min(whitespace_separators, key=lambda x: abs(x - mid_point))
            else:
                # Fallback to finding optimal separator
                optimal_separator = find_optimal_separator(content_blocks, centers, width)
            
            # Group blocks by column using the whitespace-based separator
            col1_blocks = []
            col2_blocks = []
            
            for block in content_blocks:
                block_center = block['left'] + block['width']/2
                if block_center < optimal_separator:
                    col1_blocks.append(block)
                else:
                    col2_blocks.append(block)
            
            # Add column data to JSON
            column_data = [
                {
                    "column_index": 0,
                    "center": float(centers[0]),
                    "blocks": [{
                        "text": block['text'],
                        "coords": [float(block['left']), float(block['top']), 
                                  float(block['left'] + block['width']), float(block['top'] + block['height'])]
                    } for block in col1_blocks]
                },
                {
                    "column_index": 1,
                    "center": float(centers[1]),
                    "blocks": [{
                        "text": block['text'],
                        "coords": [float(block['left']), float(block['top']), 
                                  float(block['left'] + block['width']), float(block['top'] + block['height'])]
                    } for block in col2_blocks]
                }
            ]
        elif layout_type == "Single-column":
            # Add all content blocks to a single column
            column_data = [
                {
                    "column_index": 0,
                    "center": float(centers[0]) if centers else float(width/2),
                    "blocks": [{
                        "text": block['text'],
                        "coords": [float(block['left']), float(block['top']), 
                                  float(block['left'] + block['width']), float(block['top'] + block['height'])]
                    } for block in content_blocks]
                }
            ]
        
        # Store the results
        page_results.append({
            "page": page_num,
            "layout": layout_type,
            "header_blocks": len(header_blocks),
            "footer_blocks": len(footer_blocks),
            "content_blocks": len(content_blocks)
        })
        
        # Store page data for JSON output
        page_data = {
            "page_number": page_num,
            "layout": layout_type,
            "header": header_data,
            "footer": footer_data,
            "tables": table_data,
            "columns": column_data
        }
        json_result["pages"].append(page_data)
        
       
        # Then modify your JSON saving code to use the custom encoder
        if save_json:
            json_output_path = os.path.join(output_dir, f"{pdf_name}_analysis.json")
            with open(json_output_path, 'w') as json_file:
                json.dump(json_result, json_file, indent=4, cls=NumpyEncoder)
            print(f"JSON analysis saved to: {json_output_path}")
            
        print(f"  Final classification: {layout_type}")
        print(f"  Header blocks: {len(header_blocks)}")
        print(f"  Footer blocks: {len(footer_blocks)}")
        print(f"  Content blocks: {len(content_blocks)}")
        print(f"  Tables found: {len(table_data)}")
        
        # STAGE 6: Visualization
        # ----------------------
        
        # Create enhanced visualization
        fig = create_enhanced_visualization(
            img_rgb, content_blocks, header_blocks, footer_blocks, table_boxes,
            layout_type, centers, whitespace_profile, None, None, None
        )

        # Save the visualization
        output_path = os.path.join(output_dir, f"{pdf_name}_page{page_num}_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Enhanced visualization saved to: {output_path}")

    
    # Summarize results
    single_count = sum(1 for p in page_results if p["layout"].startswith("Single"))
    dual_count = sum(1 for p in page_results if p["layout"] == "Dual-column")
    possible_dual_count = sum(1 for p in page_results if p["layout"] == "Possible dual-column")
    none_count = sum(1 for p in page_results if p["layout"] == "No text detected")
    
    print("\nSummary:")
    print("-" * 50)
    print(f"PDF: {pdf_name}")
    print(f"Total pages: {len(images)}")
    print(f"Single-column pages: {single_count}")
    print(f"Dual-column pages: {dual_count}")
    print(f"Possible dual-column pages: {possible_dual_count}")
    print(f"Pages with no text detected: {none_count}")
    
    # Determine overall classification with increased confidence
    weighted_dual_score = dual_count + (possible_dual_count * 0.5)
    if weighted_dual_score > single_count:
        overall = "DUAL-COLUMN PDF"
    else:
        overall = "SINGLE-COLUMN PDF"
        
    print(f"Overall classification: {overall}")
    
    return {
        "filename": pdf_name,
        "total_pages": len(images),
        "single_column_pages": single_count,
        "dual_column_pages": dual_count,
        "possible_dual_column_pages": possible_dual_count,
        "no_text_pages": none_count,
        "page_results": page_results,
        "overall_classification": overall
    }
    
def batch_analyze_pdfs(directories):
    results = {}
    
    for dir_type, directory in directories.items():
        print(f"\nProcessing {dir_type} directory: {directory}")
        print("=" * 60)
        
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue
            
        pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            continue
            
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            results[pdf_file] = combined_pdf_layout_analysis(pdf_file)
    
    return results
# Run the analysis
sample_dirs = {
    #  "single_column": "./samples/single_col",
     "dual_column": "./samples/dual_col"
}
    
# Example usage
if __name__ == "__main__":
    # Analyze a single PDF
    result = combined_pdf_layout_analysis("./samples/dual_col/6675433.pdf")
    
    # This function could be extended with batch processing capabilities
    # from the second algorithm if needed
    
    # batch_results = batch_analyze_pdfs(sample_dirs)