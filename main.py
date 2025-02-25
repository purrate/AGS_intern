import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def analyze_pdf_columns(pdf_path):
    # Create output directories if they don't exist
    output_dir = os.path.join(os.path.dirname(pdf_path), "column_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    
    # Process each page
    results = []
    
    print(f"\nAnalyzing PDF: {pdf_name}")
    print("-" * 50)
    
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"Processing page {page_num} of {len(images)}...")
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # OCR with Tesseract to get positional data
        data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Extract x-coordinates of text blocks
        x_coords = [data['left'][i] for i in range(len(data['text'])) 
                   if data['text'][i].strip() != '']
        
        # Analyze distribution of x-coordinates
        if not x_coords:
            result = "No text detected"
            results.append(result)
            print(f"  Page {page_num}: {result}")
            continue
            
        # Simple histogram analysis
        hist, bins = np.histogram(x_coords, bins=20)
        peaks = np.where(hist > np.max(hist) * 0.5)[0]
        
        # If we have multiple distinct peaks in x-coordinate distribution, likely multi-column
        if len(peaks) >= 2 and np.max(np.diff(peaks)) > 2:
            result = "Dual-column layout detected"
        else:
            result = "Single-column layout detected"
            
        results.append(result)
        print(f"  Page {page_num}: {result}")
        
        # Plot the distribution for visualization
        plt.figure(figsize=(10, 4))
        plt.hist(x_coords, bins=20)
        plt.title(f"{pdf_name} - Page {page_num} - {result}")
        plt.xlabel("X-coordinate")
        plt.ylabel("Frequency")
        
        # Save the histogram
        output_file = os.path.join(output_dir, f"{pdf_name}_page{page_num}_histogram.png")
        plt.savefig(output_file)
        plt.close()
        
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    
    single_count = results.count("Single-column layout detected")
    dual_count = results.count("Dual-column layout detected")
    none_count = results.count("No text detected")
    
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
    print("-" * 50)
    
    return {
        "filename": pdf_name,
        "total_pages": len(images),
        "single_column_pages": single_count,
        "dual_column_pages": dual_count,
        "no_text_pages": none_count,
        "page_results": results,
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
            results[pdf_file] = analyze_pdf_columns(pdf_file)
    
    return results

# Run the analysis
sample_dirs = {
     "single_column": "./samples/single_col/8_7973018.pdf",
    # "dual_column": "./samples/dual_col"
}

# Analyze individual PDFs
print("Individual PDF Analysis")
print("=" * 60)
result1 = analyze_pdf_columns("./samples/single_col/8_7973018.pdf")  # Single column
# result2 = analyze_pdf_columns("./samples/dual_col/637847713.pdf")  # Dual column

# Uncomment to analyze all PDFs in the directories
# print("\nBatch Analysis")
# print("=" * 60)
# batch_results = batch_analyze_pdfs(sample_dirs)