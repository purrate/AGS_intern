import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from collections import Counter

def combined_pdf_layout_analysis(pdf_path, output_dir=None):
    """
    Combined algorithm that leverages strengths of both approaches:
    1. First uses histogram analysis for initial classification
    2. Then applies appropriate detailed analysis based on classification
    
    Parameters:
    - pdf_path: Path to the PDF file
    - output_dir: Optional custom output directory
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
    
    # Process each page
    page_results = []
    
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"\nProcessing page {page_num} of {len(images)}...")
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # For matplotlib
        height, width = img_cv.shape[:2]
        
        # STAGE 1: Initial column classification using histogram method
        # -------------------------------------------------------------
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Extract x-coordinates of text blocks
        x_coords = [data['left'][j] for j in range(len(data['text'])) 
                   if data['text'][j].strip() != '']
        
        # Analyze distribution of x-coordinates
        if not x_coords:
            result = "No text detected"
            page_results.append({"page": page_num, "layout": result})
            print(f"  Page {page_num}: {result}")
            continue
            
        # Simple histogram analysis
        hist, bins = np.histogram(x_coords, bins=20)
        peaks = np.where(hist > np.max(hist) * 0.5)[0]
        
        # Initial classification
        if len(peaks) >= 2 and np.max(np.diff(peaks)) > 2:
            initial_layout = "dual-column"
        else:
            initial_layout = "single-column"
            
        print(f"  Initial classification: {initial_layout}")
        
        # STAGE 2: Extract text blocks for detailed analysis
        # --------------------------------------------------
        text_blocks = []
        for j in range(len(data['text'])):
            if data['text'][j].strip() != '' and data['conf'][j] > 30:  # Filter low confidence
                text_blocks.append({
                    'text': data['text'][j],
                    'left': data['left'][j],
                    'top': data['top'][j],
                    'width': data['width'][j],
                    'height': data['height'][j],
                    'conf': data['conf'][j]
                })
        
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
        
        # STAGE 4: Detailed column analysis based on initial classification
        # ----------------------------------------------------------------
        columns = []
        centers = []
        layout_type = initial_layout
        
        # For dual-column layouts, use K-means for more accurate column boundaries
        if initial_layout == "dual-column" and len(content_blocks) > 5:
            # Get x midpoints of text blocks
            x_midpoints = np.array([block['left'] + block['width']/2 
                                   for block in content_blocks]).reshape(-1, 1)
            
            # Apply K-means with k=2
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x_midpoints)
            
            # Check if clusters are well-separated
            centers = sorted(kmeans.cluster_centers_.flatten())
            separation = centers[1] - centers[0]
            
            # Confirm dual-column only if centers are well-separated
            if separation > width * 0.25:
                layout_type = "Dual-column"
                
                # Get cluster assignments
                labels = kmeans.labels_
                
                # Group text blocks by column
                col1_blocks = [content_blocks[i] for i in range(len(content_blocks)) if labels[i] == 0]
                col2_blocks = [content_blocks[i] for i in range(len(content_blocks)) if labels[i] == 1]
                
                # Sort columns by x position
                if col1_blocks and col2_blocks:
                    col1_mean_x = np.mean([b['left'] for b in col1_blocks])
                    col2_mean_x = np.mean([b['left'] for b in col2_blocks])
                    
                    if col1_mean_x > col2_mean_x:
                        col1_blocks, col2_blocks = col2_blocks, col1_blocks
                        
                columns = [col1_blocks, col2_blocks]
            else:
                # Revert to single-column if separation isn't enough
                layout_type = "Single-column (revised)"
                columns = [content_blocks]
                centers = [np.mean(x_midpoints)] if len(x_midpoints) > 0 else []
        else:
            # For single-column layouts, use simple approach
            layout_type = "Single-column"
            columns = [content_blocks]
            x_centers = [block['left'] + (block['width'] / 2) for block in content_blocks]
            centers = [np.mean(x_centers)] if x_centers else []
        
        # Store the results
        page_results.append({
            "page": page_num,
            "layout": layout_type,
            "header_blocks": len(header_blocks),
            "footer_blocks": len(footer_blocks),
            "content_blocks": len(content_blocks)
        })
        
        print(f"  Final classification: {layout_type}")
        print(f"  Header blocks: {len(header_blocks)}")
        print(f"  Footer blocks: {len(footer_blocks)}")
        print(f"  Content blocks: {len(content_blocks)}")
        
        # STAGE 5: Visualization
        # ----------------------
        # Create detailed visualization only for interesting layouts (dual-column or edge cases)
        if layout_type.startswith("Dual") or initial_layout != layout_type:
            # Create the visualization (similar to first algorithm)
            fig = plt.figure(figsize=(15, 10))
            
            # Original image with layout overlay
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(img_rgb)
            ax1.set_title(f"Layout Analysis - Page {page_num}")
            
            # Draw header region
            header_rect = patches.Rectangle((0, 0), width, header_height, 
                                          linewidth=2, edgecolor='r', facecolor='r', alpha=0.1)
            ax1.add_patch(header_rect)
            
            # Draw footer region
            footer_rect = patches.Rectangle((0, footer_start), width, height-footer_start, 
                                          linewidth=2, edgecolor='r', facecolor='r', alpha=0.1)
            ax1.add_patch(footer_rect)
            
            # Draw columns
            if layout_type == "Dual-column" and len(centers) == 2:
                # Estimate column widths
                col_width = width * 0.35  # Approximate width for visualization
                
                # Left column
                left_col = patches.Rectangle((centers[0] - col_width/2, header_height), 
                                          col_width, footer_start-header_height, 
                                          linewidth=2, edgecolor='g', facecolor='g', alpha=0.1)
                ax1.add_patch(left_col)
                
                # Right column
                right_col = patches.Rectangle((centers[1] - col_width/2, header_height), 
                                           col_width, footer_start-header_height, 
                                           linewidth=2, edgecolor='g', facecolor='g', alpha=0.1)
                ax1.add_patch(right_col)
            else:
                # Full width content area with margins
                margin = width * 0.15
                content_rect = patches.Rectangle((margin, header_height), 
                                              width - 2*margin, footer_start-header_height, 
                                              linewidth=2, edgecolor='g', facecolor='g', alpha=0.1)
                ax1.add_patch(content_rect)
            
            # Add layout type info
            ax1.text(10, height-10, f"Layout: {layout_type}", 
                    color='black', fontsize=12, backgroundcolor='white')
            
            # Text block visualization
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(np.ones_like(img_rgb) * 255)  # White background
            ax2.set_title(f"Text Block Detection - Page {page_num}")
            
            # Draw text blocks by category
            for block in header_blocks:
                rect = patches.Rectangle((block['left'], block['top']), 
                                      block['width'], block['height'], 
                                      linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
                ax2.add_patch(rect)
                
            for block in footer_blocks:
                rect = patches.Rectangle((block['left'], block['top']), 
                                      block['width'], block['height'], 
                                      linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
                ax2.add_patch(rect)
                
            # Color code content blocks by column
            if layout_type == "Dual-column" and len(columns) == 2:
                for i, col_blocks in enumerate(columns):
                    color = 'blue' if i == 0 else 'green'
                    for block in col_blocks:
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
            
            # X-coordinate histogram
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_title(f"X-coordinate Distribution - Page {page_num}")
            
            if content_blocks:
                x_centers = [block['left'] + (block['width'] / 2) for block in content_blocks]
                ax3.hist(x_centers, bins=30, color='blue', alpha=0.7)
                ax3.set_xlabel("X-coordinate")
                ax3.set_ylabel("Frequency")
                
                # Add vertical lines for detected column centers
                for center in centers:
                    ax3.axvline(x=center, color='red', linestyle='--', linewidth=2)
            
            # Initial vs. Final classification
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.set_title("Classification Analysis")
            ax4.axis('off')  # Hide axes
            
            # Text explanation
            ax4.text(0.5, 0.9, "Classification Summary", 
                    fontsize=12, fontweight='bold', ha='center')
            ax4.text(0.1, 0.8, f"Initial layout (histogram): {initial_layout}", fontsize=11)
            ax4.text(0.1, 0.7, f"Final layout (refined): {layout_type}", fontsize=11)
            
            if initial_layout != layout_type:
                ax4.text(0.1, 0.6, "Note: Classification was revised after detailed analysis", 
                        fontsize=10, fontstyle='italic')
                ax4.text(0.1, 0.5, f"Reason: {'Insufficient column separation' if layout_type == 'Single-column (revised)' else 'K-means detected clear columns'}", 
                        fontsize=10)
            
            ax4.text(0.1, 0.4, f"Content blocks: {len(content_blocks)}", fontsize=10)
            ax4.text(0.1, 0.3, f"Header blocks: {len(header_blocks)}", fontsize=10)
            ax4.text(0.1, 0.2, f"Footer blocks: {len(footer_blocks)}", fontsize=10)
            
            plt.tight_layout()
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"{pdf_name}_page{page_num}_analysis.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Detailed visualization saved to: {output_path}")
        else:
            # For simple single-column layouts, just save a basic histogram
            plt.figure(figsize=(8, 4))
            plt.hist(x_coords, bins=20)
            plt.title(f"{pdf_name} - Page {page_num} - {layout_type}")
            plt.xlabel("X-coordinate")
            plt.ylabel("Frequency")
            
            output_path = os.path.join(output_dir, f"{pdf_name}_page{page_num}_basic.png")
            plt.savefig(output_path)
            plt.close()
            print(f"  Basic visualization saved to: {output_path}")
    
    # Summarize results
    single_count = sum(1 for p in page_results if p["layout"].startswith("Single"))
    dual_count = sum(1 for p in page_results if p["layout"].startswith("Dual"))
    none_count = sum(1 for p in page_results if p["layout"] == "No text detected")
    
    print("\nSummary:")
    print("-" * 50)
    print(f"PDF: {pdf_name}")
    print(f"Total pages: {len(images)}")
    print(f"Single-column pages: {single_count}")
    print(f"Dual-column pages: {dual_count}")
    print(f"Pages with no text detected: {none_count}")
    
    # Determine overall classification
    if dual_count > single_count:
        overall = "DUAL-COLUMN PDF"
    else:
        overall = "SINGLE-COLUMN PDF"
        
    print(f"Overall classification: {overall}")
    
    return {
        "filename": pdf_name,
        "total_pages": len(images),
        "single_column_pages": single_count,
        "dual_column_pages": dual_count,
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
     "single_column": "./samples/single_col/8_7973018.pdf",
    # "dual_column": "./samples/dual_col"
}

# Example usage
if __name__ == "__main__":
    # Analyze a single PDF
    # result = combined_pdf_layout_analysis("./samples/dual_col/637847713.pdf")
    
    # This function could be extended with batch processing capabilities
    # from the second algorithm if needed
    
    batch_results = batch_analyze_pdfs(sample_dirs)