import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from collections import Counter
from sklearn.metrics import silhouette_score
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def combined_pdf_layout_analysis(pdf_path, output_dir=None):
    """
    Combined algorithm that leverages strengths of both approaches:
    1. First uses histogram analysis for initial classification
    2. Then applies appropriate detailed analysis based on classification
    3. Includes table detection without sacrificing dual-column detection
    
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
        
        # STAGE 1: OCR to extract text blocks
        # -------------------------------------------------------------
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Extract text blocks
        text_blocks = []
        for j in range(len(data['text'])):
            if data['text'][j].strip() != '' and data['conf'][j] > 30:  # Filter low confidence
                text_blocks.append({
                    'text': data['text'][j],
                    'left': data['left'][j],
                    'top': data['top'][j],
                    'width': data['width'][j],
                    'height': data['height'][j],
                    'conf': data['conf'][j],
                    'line_num': data['line_num'][j],
                    'block_num': data['block_num'][j]
                })
        
        # If no text is detected, continue to next page
        if not text_blocks:
            result = "No text detected"
            page_results.append({"page": page_num, "layout": result})
            print(f"  Page {page_num}: {result}")
            continue
        
        # STAGE 2: Table detection 
        # -------------------------------------------------------------
        # Group blocks by line_num for line analysis
        lines = {}
        for block in text_blocks:
            line_key = f"{block['block_num']}_{block['line_num']}"
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(block)
        
        # Analyze lines to detect potential tables
        table_likelihood = 0
        table_blocks = []
        line_block_counts = []
        
        for line_key, blocks in lines.items():
            if len(blocks) >= 3:  # Lines with multiple text blocks might indicate tables
                line_block_counts.append(len(blocks))
                
                # Check horizontal alignment - tables often have aligned columns
                lefts = [block['left'] for block in blocks]
                lefts_sorted = sorted(lefts)
                if len(lefts) >= 3:
                    # Calculate differences between consecutive x positions
                    diffs = [lefts_sorted[i+1] - lefts_sorted[i] for i in range(len(lefts_sorted)-1)]
                    # If differences are somewhat consistent, likely a table
                    if np.std(diffs) < np.mean(diffs) * 0.5:
                        table_likelihood += 1
                        table_blocks.extend(blocks)
        
        # Determine if the page likely contains tables
        has_tables = False
        table_coverage = 0
        if line_block_counts:
            avg_blocks_per_line = sum(line_block_counts) / len(line_block_counts)
            if avg_blocks_per_line > 2.5 and table_likelihood > 2:
                has_tables = True
                # Calculate approximate table coverage (percentage of content blocks)
                table_coverage = len(table_blocks) / len(text_blocks) * 100 if text_blocks else 0
                print(f"  Table detected on page {page_num} (likelihood score: {table_likelihood}, coverage: {table_coverage:.1f}%)")
        
        # STAGE 3: Initial column classification using histogram method
        # -------------------------------------------------------------
        # Extract x-coordinates of text blocks (midpoints for better representation)
        x_midpoints = [block['left'] + block['width']/2 for block in text_blocks]
        
        # Simple histogram analysis
        hist, bins = np.histogram(x_midpoints, bins=20)
        normalized_hist = hist / np.max(hist) if np.max(hist) > 0 else hist
        peaks = np.where(normalized_hist > 0.5)[0]
        
        # Filter peaks to ensure they're well-separated
        if len(peaks) >= 2:
            # Convert bin indices to x-coordinates
            peak_positions = [(bins[p] + bins[p+1])/2 for p in peaks]
            
            # Calculate distances between peaks
            peak_diffs = np.diff(peak_positions)
            
            # Only count peaks that are significantly separated (at least 20% of page width)
            significant_peaks = 1  # Start with 1 for the first peak
            for diff in peak_diffs:
                if diff > width * 0.2:
                    significant_peaks += 1
            
            # Update peaks count to significant peaks only
            num_peaks = significant_peaks
        else:
            num_peaks = len(peaks)
        
        # Initial classification
        if num_peaks >= 2:
            initial_layout = "dual-column"
        else:
            initial_layout = "single-column"
            
        print(f"  Initial classification: {initial_layout} (peaks: {num_peaks})")
        
        # STAGE 4: Header/Footer detection
        # --------------------------------
        # Simple header/footer estimate (10% of page)
        header_height = int(height * 0.1)
        footer_start = height - int(height * 0.1)
        
        # Separate text blocks by region
        header_blocks = [block for block in text_blocks if block['top'] < header_height]
        footer_blocks = [block for block in text_blocks if block['top'] > footer_start]
        content_blocks = [block for block in text_blocks 
                         if block['top'] >= header_height and block['top'] <= footer_start]
        
        # Separate table blocks from content blocks for better column analysis
        non_table_blocks = []
        if has_tables:
            table_block_ids = [(block['block_num'], block['line_num']) for block in table_blocks]
            non_table_blocks = [block for block in content_blocks 
                               if (block['block_num'], block['line_num']) not in table_block_ids]
        else:
            non_table_blocks = content_blocks
        
        # STAGE 5: Detailed column analysis using K-means clustering
        # ----------------------------------------------------------------
        columns = []
        centers = []
        layout_type = initial_layout
        
        # Apply K-means clustering to non-table blocks if there are enough
        significant_blocks = non_table_blocks if has_tables and len(non_table_blocks) > 10 else content_blocks
        
        if len(significant_blocks) > 5:
            # Get x midpoints of significant text blocks
            x_midpoints_array = np.array([block['left'] + block['width']/2 
                                        for block in significant_blocks]).reshape(-1, 1)
            
            # Try clustering with K=2 first
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x_midpoints_array)
            
            # Calculate clustering quality metrics
            centers = sorted(kmeans.cluster_centers_.flatten())
            separation = centers[1] - centers[0] if len(centers) >= 2 else 0
            
            # Calculate silhouette score if possible
            sil_score = 0
            if len(x_midpoints_array) >= 4:  # Need at least 4 points for meaningful silhouette score
                sil_score = silhouette_score(x_midpoints_array, kmeans.labels_)
                print(f"  Silhouette score: {sil_score:.3f}, column separation: {separation/width:.2f} of page width")
            
            # Determine layout type based on clustering quality
            if len(centers) >= 2 and separation > width * 0.25 and sil_score > 0.5:
                layout_type = "Dual-column"
                
                # Get cluster assignments
                labels = kmeans.labels_
                
                # Group text blocks by column
                col1_blocks = [significant_blocks[i] for i in range(len(significant_blocks)) if labels[i] == 0]
                col2_blocks = [significant_blocks[i] for i in range(len(significant_blocks)) if labels[i] == 1]
                
                # Sort columns by x position
                if col1_blocks and col2_blocks:
                    col1_mean_x = np.mean([b['left'] for b in col1_blocks])
                    col2_mean_x = np.mean([b['left'] for b in col2_blocks])
                    
                    if col1_mean_x > col2_mean_x:
                        col1_blocks, col2_blocks = col2_blocks, col1_blocks
                        
                columns = [col1_blocks, col2_blocks]
            else:
                layout_type = "Single-column"
                if has_tables:
                    if table_coverage > 40:  # If tables cover a significant portion
                        layout_type = "Single-column with tables"
                    elif separation > width * 0.25 and sil_score > 0.4:
                        # If we see clear column structure despite some tables,
                        # it might be a dual-column document with tables
                        layout_type = "Dual-column with tables"
                
                columns = [significant_blocks]
                centers = [np.mean(x_midpoints_array)] if len(x_midpoints_array) > 0 else []
        else:
            # For insufficient blocks, use simple approach
            layout_type = "Single-column"
            if has_tables:
                layout_type = "Single-column with tables"
            columns = [significant_blocks]
            x_centers = [block['left'] + (block['width'] / 2) for block in significant_blocks]
            centers = [np.mean(x_centers)] if x_centers else []
        
        # Store the results
        page_results.append({
            "page": page_num,
            "layout": layout_type,
            "header_blocks": len(header_blocks),
            "footer_blocks": len(footer_blocks),
            "content_blocks": len(content_blocks),
            "has_tables": has_tables,
            "table_coverage": table_coverage if has_tables else 0
        })
        
        print(f"  Final classification: {layout_type}")
        print(f"  Header blocks: {len(header_blocks)}")
        print(f"  Footer blocks: {len(footer_blocks)}")
        print(f"  Content blocks: {len(content_blocks)}")
        
        # STAGE 6: Visualization
        # ----------------------
        # Create detailed visualization
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
        if layout_type.startswith("Dual-column") and len(centers) == 2:
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
            
        # Color code content blocks based on their type
        if has_tables:
            # Draw table blocks in purple
            for block in table_blocks:
                if header_height <= block['top'] <= footer_start:  # Only content area
                    rect = patches.Rectangle((block['left'], block['top']), 
                                          block['width'], block['height'], 
                                          linewidth=1, edgecolor='purple', facecolor='purple', alpha=0.3)
                    ax2.add_patch(rect)
                    
            # Draw non-table blocks based on column layout
            if layout_type.startswith("Dual-column") and len(columns) == 2:
                for i, col_blocks in enumerate(columns):
                    color = 'blue' if i == 0 else 'green'
                    for block in col_blocks:
                        # Skip if already drawn as a table block
                        if (block['block_num'], block['line_num']) not in [(b['block_num'], b['line_num']) for b in table_blocks]:
                            rect = patches.Rectangle((block['left'], block['top']), 
                                                  block['width'], block['height'], 
                                                  linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
                            ax2.add_patch(rect)
            else:
                # For non-table blocks in single column
                for block in non_table_blocks:
                    rect = patches.Rectangle((block['left'], block['top']), 
                                          block['width'], block['height'], 
                                          linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
                    ax2.add_patch(rect)
        elif layout_type.startswith("Dual-column") and len(columns) == 2:
            # Draw dual column without tables
            for i, col_blocks in enumerate(columns):
                color = 'blue' if i == 0 else 'green'
                for block in col_blocks:
                    rect = patches.Rectangle((block['left'], block['top']), 
                                          block['width'], block['height'], 
                                          linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
                    ax2.add_patch(rect)
        else:
            # Single column without tables
            for block in content_blocks:
                rect = patches.Rectangle((block['left'], block['top']), 
                                      block['width'], block['height'], 
                                      linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
                ax2.add_patch(rect)
        
        # X-coordinate histogram
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title(f"X-coordinate Distribution - Page {page_num}")
        
        if content_blocks:
            # Plot histograms separately for table and non-table blocks if tables exist
            if has_tables and non_table_blocks:
                # Non-table blocks
                non_table_x = [block['left'] + (block['width'] / 2) for block in non_table_blocks]
                ax3.hist(non_table_x, bins=30, color='blue', alpha=0.7, label='Non-table text')
                
                # Table blocks
                table_x = [block['left'] + (block['width'] / 2) for block in table_blocks]
                ax3.hist(table_x, bins=30, color='purple', alpha=0.3, label='Table text')
                
                ax3.legend()
            else:
                # All blocks
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
        ax4.text(0.5, 0.95, "Classification Summary", 
                fontsize=12, fontweight='bold', ha='center')
        ax4.text(0.1, 0.85, f"Initial layout (histogram): {initial_layout}", fontsize=11)
        ax4.text(0.1, 0.78, f"Final layout (refined): {layout_type}", fontsize=11)
        
        # Table information
        row_pos = 0.71
        if has_tables:
            ax4.text(0.1, row_pos, f"Tables detected: Yes (coverage: {table_coverage:.1f}%)", 
                    fontsize=10, color='purple')
            row_pos -= 0.07
            ax4.text(0.1, row_pos, f"Table likelihood score: {table_likelihood}", 
                    fontsize=10, color='purple')
            row_pos -= 0.07
        
        # Column information
        if len(centers) >= 2:
            ax4.text(0.1, row_pos, f"Column separation: {separation/width:.2f} of page width", 
                    fontsize=10)
            row_pos -= 0.07
            if layout_type.startswith("Dual"):
                ax4.text(0.1, row_pos, "Column quality: Good", fontsize=10, color='green')
            else:
                ax4.text(0.1, row_pos, "Column quality: Insufficient", fontsize=10, color='red')
            row_pos -= 0.07
        
        # Block counts
        ax4.text(0.1, row_pos, f"Content blocks: {len(content_blocks)}", fontsize=10)
        row_pos -= 0.07
        ax4.text(0.1, row_pos, f"Header blocks: {len(header_blocks)}", fontsize=10)
        row_pos -= 0.07
        ax4.text(0.1, row_pos, f"Footer blocks: {len(footer_blocks)}", fontsize=10)
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"{pdf_name}_page{page_num}_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Visualization saved to: {output_path}")
    
    # Summarize results
    single_count = sum(1 for p in page_results if p["layout"].startswith("Single"))
    dual_count = sum(1 for p in page_results if p["layout"].startswith("Dual"))
    table_count = sum(1 for p in page_results if p.get("has_tables", False))
    none_count = sum(1 for p in page_results if p["layout"] == "No text detected")
    
    print("\nSummary:")
    print("-" * 50)
    print(f"PDF: {pdf_name}")
    print(f"Total pages: {len(images)}")
    print(f"Single-column pages: {single_count}")
    print(f"Dual-column pages: {dual_count}")
    print(f"Pages with tables: {table_count}")
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
        "table_pages": table_count,
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
     "single_column": os.path.join(os.path.dirname(__file__), "samples", "single_col" ),
    # "dual_column": "./samples/dual_col"
}

# Example usage
if __name__ == "__main__":
    # Analyze a single PDF
    # result = combined_pdf_layout_analysis("./samples/dual_col/637847713.pdf")
    
    # Or use batch processing
    batch_results = batch_analyze_pdfs(sample_dirs)