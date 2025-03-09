import numpy as np
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d



def calculate_layout_features(content_blocks, header_blocks, footer_blocks, table_boxes, 
                            width, height, whitespace_profile=None):
    """
    Calculates features for more robust layout classification.
    """
    if not content_blocks:
        return {"empty_page": True}
    
    # Get basic block statistics
    x_centers = [block['left'] + block['width']/2 for block in content_blocks]
    y_centers = [block['top'] + block['height']/2 for block in content_blocks]
    widths = [block['width'] for block in content_blocks]
    heights = [block['height'] for block in content_blocks]
    
    # Calculate feature 1: Horizontal distribution of text
    hist, _ = np.histogram(x_centers, bins=10, range=(0, width))
    hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
    # Calculate variance and peaks in distribution
    x_variance = np.var(x_centers) / (width ** 2)  # Normalized variance
    
    # Use smoothed histogram
    hist_smooth = gaussian_filter1d(hist_normalized, sigma=1)
    peaks, _ = find_peaks(hist_smooth, height=0.1, distance=2)
    
    # Calculate feature 2: Text block density
    text_area = sum(w * h for w, h in zip(widths, heights))
    page_area = width * height
    text_density = text_area / page_area
    
    # Calculate feature 3: Gap features from whitespace
    gap_features = {}
    if whitespace_profile is not None:
        # Find large gaps in whitespace profile
        gaps = []
        threshold = 0.9
        in_gap = False
        start = 0
        
        for i, val in enumerate(whitespace_profile):
            if val > threshold and not in_gap:
                in_gap = True
                start = i
            elif val <= threshold and in_gap:
                in_gap = False
                gap_width = i - start
                if gap_width > width * 0.05:  # Only consider gaps wider than 5% of page
                    gaps.append((start, i, gap_width))
        
        # End case
        if in_gap:
            gap_width = len(whitespace_profile) - start
            if gap_width > width * 0.05:
                gaps.append((start, len(whitespace_profile), gap_width))
        
        # Gap statistics
        gap_features = {
            "num_gaps": len(gaps),
            "max_gap_width": max([g[2] for g in gaps]) / width if gaps else 0,
            "total_gap_width": sum([g[2] for g in gaps]) / width if gaps else 0,
            "gap_positions": [g[0]/width for g in gaps]  # Normalized positions
        }
    
    # Calculate feature 4: Column alignment
    alignment_features = {}
    if len(content_blocks) > 5:
        # Try to find column edges
        left_edges = [block['left'] for block in content_blocks]
        right_edges = [block['left'] + block['width'] for block in content_blocks]
        
        # Cluster left edges
        X_left = np.array(left_edges).reshape(-1, 1)
        left_db = DBSCAN(eps=width*0.05, min_samples=3).fit(X_left)
        left_labels = set(left_db.labels_) - {-1}
        
        # Measure consistency within each cluster
        left_consistency = []
        for label in left_labels:
            edges = X_left[left_db.labels_ == label]
            left_consistency.append(1 - (np.std(edges) / width))
        
        # Similar for right edges
        X_right = np.array(right_edges).reshape(-1, 1)
        right_db = DBSCAN(eps=width*0.05, min_samples=3).fit(X_right)
        right_labels = set(right_db.labels_) - {-1}
        
        right_consistency = []
        for label in right_labels:
            edges = X_right[right_db.labels_ == label]
            right_consistency.append(1 - (np.std(edges) / width))
        
        alignment_features = {
            "num_left_edges": len(left_labels),
            "num_right_edges": len(right_labels),
            "max_left_consistency": max(left_consistency) if left_consistency else 0,
            "max_right_consistency": max(right_consistency) if right_consistency else 0,
            "avg_left_consistency": np.mean(left_consistency) if left_consistency else 0,
            "avg_right_consistency": np.mean(right_consistency) if right_consistency else 0
        }
    
    # Combine all features
    features = {
        "block_count": len(content_blocks),
        "header_count": len(header_blocks),
        "footer_count": len(footer_blocks),
        "table_count": len(table_boxes),
        "x_variance": x_variance,
        "peak_count": len(peaks),
        "text_density": text_density,
        "empty_page": False,
        **gap_features,
        **alignment_features
    }
    
    return features

def classify_layout_with_features(features):
    """
    Classifies the layout using calculated features and rule-based decision.
    """
    # Simple rule-based classifier based on features
    if features.get("empty_page", True):
        return "No text detected", 1.0
        
    # Calculate dual-column evidence
    dual_col_evidence = 0
    
    # Evidence from horizontal distribution
    if features.get("peak_count", 0) >= 2:
        dual_col_evidence += 2
    
    # Evidence from gaps
    if features.get("num_gaps", 0) >= 1 and features.get("max_gap_width", 0) > 0.1:
        dual_col_evidence += 3
        
    # Evidence from alignment
    if features.get("num_left_edges", 0) >= 2 and features.get("num_right_edges", 0) >= 2:
        dual_col_evidence += 2
    
    # Extra evidence from strong alignment
    if features.get("max_left_consistency", 0) > 0.85 and features.get("max_right_consistency", 0) > 0.85:
        dual_col_evidence += 2
        
    # Calculate single-column evidence
    single_col_evidence = 0
    
    # Evidence from horizontal distribution
    if features.get("peak_count", 0) <= 1:
        single_col_evidence += 2
        
    # Evidence from gaps
    if features.get("num_gaps", 0) == 0 or features.get("max_gap_width", 0) < 0.05:
        single_col_evidence += 2
        
    # Evidence from alignment
    if features.get("num_left_edges", 0) <= 1 and features.get("num_right_edges", 0) <= 1:
        single_col_evidence += 2
    
    # Make classification with confidence
    if dual_col_evidence > single_col_evidence + 2:
        confidence = min(0.95, 0.6 + (dual_col_evidence - single_col_evidence) * 0.05)
        return "Dual-column", confidence
    elif single_col_evidence > dual_col_evidence + 2:
        confidence = min(0.95, 0.6 + (single_col_evidence - dual_col_evidence) * 0.05)
        return "Single-column", confidence
    elif dual_col_evidence > single_col_evidence:
        return "Possible dual-column", 0.6
    else:
        return "Single-column", 0.6