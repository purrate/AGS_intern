import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d


from tables import visualize_tables
from seperator import find_optimal_separator
def create_enhanced_visualization(img_rgb, content_blocks, header_blocks, footer_blocks, 
                                 table_boxes, layout_type, centers, whitespace_profile=None, 
                                 left_margins=None, right_margins=None, adaptive_columns=None):
    """
    Creates an enhanced visualization with multiple view options.
    """
    height, width = img_rgb.shape[:2]
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Original Image with Layout Overlay
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Original with Layout")
    
    # Draw header/footer regions
    header_height = int(height * 0.1)
    footer_start = height - int(height * 0.1)
    
    header_rect = patches.Rectangle((0, 0), width, header_height, 
                                  linewidth=2, edgecolor='r', facecolor='r', alpha=0.1)
    ax1.add_patch(header_rect)
    
    footer_rect = patches.Rectangle((0, footer_start), width, height-footer_start, 
                                  linewidth=2, edgecolor='r', facecolor='r', alpha=0.1)
    ax1.add_patch(footer_rect)
    
    # Draw tables
    visualize_tables(ax1, table_boxes)
    
    # 3. Column Detection Visualization
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(np.ones_like(img_rgb) * 255)  # White background
    ax2.set_title("Column Detection")
    
    # Draw column centers
    for center in centers:
        ax2.axvline(x=center, color='green', linestyle='--', linewidth=2)
        ax2.text(center, 20, f"{center:.1f}", color='green', 
                fontsize=8, ha='center', backgroundcolor='white')
    
    # Draw adaptive column boundaries if available
    if adaptive_columns:
        for i, (left, right) in enumerate(adaptive_columns):
            rect = patches.Rectangle(
                (left, header_height), right-left, footer_start-header_height,
                linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1
            )
            ax2.add_patch(rect)
            ax2.text((left+right)/2, header_height-10, f"Col {i+1}", 
                    color='blue', fontsize=10, ha='center')
    
    # Draw text blocks color-coded by column
    if layout_type in ["Dual-column", "Possible dual-column"] and len(centers) >= 2:
        # Use optimal separator if available
        mid_separator = find_optimal_separator(content_blocks, centers, width)

        
        for block in content_blocks:
            block_center = block['left'] + block['width']/2
            color = 'blue' if block_center < mid_separator else 'green'
            rect = patches.Rectangle((block['left'], block['top']), 
                                  block['width'], block['height'], 
                                  linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
            ax2.add_patch(rect)
    else:
        for block in content_blocks:
            rect = patches.Rectangle((block['left'], block['top']), 
                                  block['width'], block['height'], 
                                  linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
            ax2.add_patch(rect)
    
    # 4. Whitespace and Margin Analysis
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.set_title("Whitespace & Margin Analysis")
    
    # Plot whitespace profile if available
    if whitespace_profile is not None:
        ax3.plot(np.arange(len(whitespace_profile)), whitespace_profile, 'b-', linewidth=1)
        ax3.set_xlim(0, len(whitespace_profile))
        ax3.set_ylim(0, 1.1)
        ax3.set_xlabel("X Position")
        ax3.set_ylabel("Whitespace Ratio")
        
        # Add threshold line
        ax3.axhline(y=0.92, color='r', linestyle='--', linewidth=1)
        ax3.text(10, 0.94, "Whitespace Threshold", color='r', fontsize=8)
        
    # Original image with layout overlay
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(img_rgb)
    ax4.set_title(f"Layout Analysis - Page")
    visualize_tables(ax4, table_boxes)
    
    # Draw header region
    header_rect = patches.Rectangle((0, 0), width, header_height, 
                                linewidth=2, edgecolor='r', facecolor='r', alpha=0.1)
    ax4.add_patch(header_rect)
    
    # Draw footer region
    footer_rect = patches.Rectangle((0, footer_start), width, height-footer_start, 
                                linewidth=2, edgecolor='r', facecolor='r', alpha=0.1)
    ax4.add_patch(footer_rect)
    
    if layout_type == "Dual-column" or "Possible dual-column" and len(centers) >= 2:
        # Use text blocks to determine actual column widths
        mid_separator = find_optimal_separator(content_blocks, centers, width)
        
        # Split blocks by column
        left_col_blocks = [block for block in content_blocks 
                        if block['left'] + block['width']/2 < mid_separator]
        right_col_blocks = [block for block in content_blocks 
                        if block['left'] + block['width']/2 >= mid_separator]
        
        # Calculate column boundaries
        if left_col_blocks:
            left_edges = [block['left'] for block in left_col_blocks]
            right_edges = [block['left'] + block['width'] for block in left_col_blocks]
            left_col_left = np.percentile(left_edges, 5) if len(left_edges) > 5 else min(left_edges)
            left_col_right = np.percentile(right_edges, 95) if len(right_edges) > 5 else max(right_edges)
            left_col_width = left_col_right - left_col_left
            buffer = width * 0.01
            
            left_col = patches.Rectangle(
                (left_col_left - buffer, header_height), 
                left_col_width + 2*buffer, footer_start-header_height, 
                linewidth=2, edgecolor='g', facecolor='g', alpha=0.1
            )
            ax4.add_patch(left_col)
        
        if right_col_blocks:
            left_edges = [block['left'] for block in right_col_blocks]
            right_edges = [block['left'] + block['width'] for block in right_col_blocks]
            right_col_left = np.percentile(left_edges, 5) if len(left_edges) > 5 else min(left_edges)
            right_col_right = np.percentile(right_edges, 95) if len(right_edges) > 5 else max(right_edges)
            right_col_width = right_col_right - right_col_left
            buffer = width * 0.01
            
            right_col = patches.Rectangle(
                (right_col_left - buffer, header_height), 
                right_col_width + 2*buffer, footer_start-header_height, 
                linewidth=2, edgecolor='g', facecolor='g', alpha=0.1
            )
            ax4.add_patch(right_col)
    else:  # Single-column layout
        # Calculate the actual content boundaries based on text blocks
        if content_blocks:
            left_margins = [block['left'] for block in content_blocks]
            right_margins = [block['left'] + block['width'] for block in content_blocks]
            
            # Get the 5th percentile for left margin and 95th percentile for right margin
            # to handle outliers while still capturing the full content width
            left_content_edge = np.percentile(left_margins, 5) if len(left_margins) > 10 else min(left_margins)
            right_content_edge = np.percentile(right_margins, 95) if len(right_margins) > 10 else max(right_margins)
            
            # Add a small buffer (e.g., 5% of page width) on each side
            buffer = width * 0.05
            content_left = max(0, left_content_edge - buffer)
            content_right = min(width, right_content_edge + buffer)
            content_width = content_right - content_left
            
            # Draw the content area rectangle
            content_rect = patches.Rectangle(
                (content_left, header_height), 
                content_width, footer_start-header_height, 
                linewidth=2, edgecolor='g', facecolor='g', alpha=0.1
            )
            ax4.add_patch(content_rect)
        else:
            # Fallback to default margins if no content blocks
            margin = width * 0.15
            content_rect = patches.Rectangle(
                (margin, header_height), 
                width - 2*margin, footer_start-header_height, 
                linewidth=2, edgecolor='g', facecolor='g', alpha=0.1
            )
            ax4.add_patch(content_rect)
    # Add layout type info
    ax4.text(10, height-10, f"Layout: {layout_type}", 
            color='black', fontsize=12, backgroundcolor='white')
    
    # Text block visualization
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(np.ones_like(img_rgb) * 255)  # White background
    ax5.set_title(f"Text Block Detection - Page")
    
    # Draw text blocks by category
    for block in header_blocks:
        rect = patches.Rectangle((block['left'], block['top']), 
                                block['width'], block['height'], 
                                linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
        ax5.add_patch(rect)
        
    for block in footer_blocks:
        rect = patches.Rectangle((block['left'], block['top']), 
                                block['width'], block['height'], 
                                linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
        ax5.add_patch(rect)
    
    # For dual-column layouts, color-code the blocks by column
    if layout_type in ["Dual-column", "Possible dual-column"] and len(centers) >= 2:
        mid_separator = (centers[0] + centers[1]) / 2
        
        for block in content_blocks:
            block_center = block['left'] + block['width']/2
            color = 'blue' if block_center < mid_separator else 'green'
            rect = patches.Rectangle((block['left'], block['top']), 
                                    block['width'], block['height'], 
                                    linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
            ax5.add_patch(rect)
    else:
        for block in content_blocks:
            rect = patches.Rectangle((block['left'], block['top']), 
                                    block['width'], block['height'], 
                                    linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
            ax5.add_patch(rect)
    
    # X-coordinate histogram
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.set_title(f"X-coordinate Distribution - Page")
    
    if content_blocks:
        x_centers = [block['left'] + (block['width'] / 2) for block in content_blocks]
        ax6.hist(x_centers, bins=30, color='blue', alpha=0.7)
        ax6.set_xlabel("X-coordinate")
        ax6.set_ylabel("Frequency")
        
        # Add vertical lines for detected column centers
        for center in centers:
            ax6.axvline(x=center, color='red', linestyle='--', linewidth=2)
            
        # Add smoothed density curve
        if len(x_centers) > 2:
            hist, bin_edges = np.histogram(x_centers, bins=50, range=(0, width))
            smoothed_hist = gaussian_filter1d(hist, sigma=1.5)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Scale the smoothed histogram to match the regular histogram
            scale_factor = ax6.get_ylim()[1] / max(smoothed_hist) if max(smoothed_hist) > 0 else 1
            ax6.plot(bin_centers, smoothed_hist * scale_factor, color='red', linewidth=2)
    
    # Classification explanation
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_title("Classification Analysis")
    ax7.axis('off')  # Hide axes
    
    # Text explanation
    ax7.text(0.5, 0.9, "Classification Summary", 
            fontsize=12, fontweight='bold', ha='center')
    ax7.text(0.1, 0.8, f"Final layout: {layout_type}", fontsize=11)
    
    if layout_type == "Possible dual-column":
        ax7.text(0.1, 0.7, "Note: Some evidence of dual columns, but not conclusive", 
                fontsize=10, fontstyle='italic')
    
    ax7.text(0.1, 0.6, f"Content blocks: {len(content_blocks)}", fontsize=10)
    ax7.text(0.1, 0.5, f"Header blocks: {len(header_blocks)}", fontsize=10)
    ax7.text(0.1, 0.4, f"Footer blocks: {len(footer_blocks)}", fontsize=10)
    
    if centers:
        ax7.text(0.1, 0.2, f"Column centers: {', '.join([f'{c:.1f}' for c in centers])}", fontsize=10)
            
    plt.tight_layout()
    return fig