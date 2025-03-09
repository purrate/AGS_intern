import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

def find_optimal_separator(content_blocks, centers, width):
    """
    Finds the optimal separation point between columns by locating the 
    whitespace gap with minimal text overlap.
    """
    if len(centers) < 2:
        return width / 2
        
    # Define a reasonable range to search for the separator
    min_x = centers[0]
    max_x = centers[1]
    search_range = np.linspace(min_x, max_x, 50)
    
    # Count text blocks that overlap with each potential separator position
    overlap_scores = []
    
    for x in search_range:
        overlap_count = 0
        for block in content_blocks:
            # Check if this potential separator intersects with the text block
            if block['left'] < x < (block['left'] + block['width']):
                overlap_count += 1
        overlap_scores.append(overlap_count)
    
    # Find the position with minimal overlap
    min_overlap_idx = np.argmin(overlap_scores)
    optimal_separator = search_range[min_overlap_idx]
    
    # If all positions have overlap, fall back to a position with minimal density
    if min(overlap_scores) > 0:
        # Create a density profile across the page width
        x_positions = []
        for block in content_blocks:
            # Add left and right edges of each block
            x_positions.append(block['left'])
            x_positions.append(block['left'] + block['width'])
            
        # Create a histogram to find areas with minimal text
        hist, bin_edges = np.histogram(x_positions, bins=100, range=(min_x, max_x))
        smoothed_hist = gaussian_filter1d(hist, sigma=2)
        
        # Find the position with minimum density between the column centers
        min_density_idx = np.argmin(smoothed_hist)
        bin_center = (bin_edges[min_density_idx] + bin_edges[min_density_idx+1]) / 2
        
        # Use this as the separator if it's between the column centers
        if min_x <= bin_center <= max_x:
            optimal_separator = bin_center
    
    return optimal_separator

def analyze_vertical_whitespace(img_gray, width, height):
    """
    Analyzes vertical whitespace to identify column gaps.
    Returns a whitespace profile and potential column separators.
    """
    # Create a binary image where white pixels are 1 and black pixels are 0
    _, binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    
    # Create vertical projection profile (sum white pixels in each column)
    v_projection = np.sum(binary == 255, axis=0) / height
    
    # Smooth the profile
    v_projection_smooth = gaussian_filter1d(v_projection, sigma=width/100)
    
    # Find runs of whitespace (continuous areas with high white pixel counts)
    whitespace_threshold = 0.92  # Consider columns with 92%+ white pixels as whitespace
    whitespace_mask = v_projection_smooth > whitespace_threshold
    
    # Find runs of whitespace using run-length encoding
    runs = []
    in_run = False
    start = 0
    
    for i, val in enumerate(whitespace_mask):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            in_run = False
            if i - start > width * 0.03:  # Only consider runs wider than 3% of page width
                runs.append((start, i))
    
    # Handle case where we end in a run
    if in_run:
        if width - start > width * 0.03:
            runs.append((start, width))
    
    # Get the centers of whitespace runs as potential column separators
    separators = [int((start + end) / 2) for start, end in runs]
    
    return v_projection_smooth, separators