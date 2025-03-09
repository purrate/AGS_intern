# PDF Layout Analysis Tool

A comprehensive Python tool for analyzing PDF document layouts, with a focus on detecting column structures, tables, headers, and footers.

## Features

- **Advanced Column Detection**: Uses multiple methods including density-based peak detection and DBSCAN clustering
- **Table Detection**: Identifies table regions within PDF documents
- **Header and Footer Extraction**: Detects and extracts text from header and footer regions
- **Layout Classification**: Classifies pages as single-column, dual-column, or possible dual-column
- **Visualization**: Creates enhanced visualizations of the document layout analysis
- **JSON Output**: Generates detailed JSON output of the analysis results

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- PyTesseract
- pdf2image
- Matplotlib
- scikit-learn
- SciPy
- Poppler (for pdf2image)

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install opencv-python numpy pytesseract pdf2image matplotlib scikit-learn scipy
```

3. Install Tesseract OCR engine:
   - For Ubuntu: `sudo apt-get install tesseract-ocr`
   - For Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - For macOS: `brew install tesseract`

4. Install Poppler for pdf2image:
   - For Ubuntu: `sudo apt-get install poppler-utils`
   - For Windows: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
   - For macOS: `brew install poppler`

## Usage

### Running the Notebook

To analyze PDF layouts using the `tabledetection.ipynb` notebook:

1. Open a terminal or command prompt and navigate to the project directory.
2. Start Jupyter Notebook or Jupyter Lab:
   
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```
3. Open `tabledetection.ipynb` in the Jupyter interface.
4. Run the notebook cells sequentially to process a PDF document and analyze its table structures.

### Single PDF Analysis (Script-Based)

```python
from pdf_layout_analyzer import combined_pdf_layout_analysis

result = combined_pdf_layout_analysis("path/to/your/document.pdf")
```

### Batch Processing

```python
from pdf_layout_analyzer import batch_analyze_pdfs

directories = {
    "single_column": "./samples/single_col",
    "dual_column": "./samples/dual_col"
}

results = batch_analyze_pdfs(directories)
```

## Output

The tool produces:

1. **JSON files** containing detailed analysis of each page, including:
   - Overall layout classification
   - Header and footer text
   - Table locations and content
   - Column information with text blocks

2. **Visualization images** for each page showing:
   - Detected text blocks
   - Column structure
   - Tables, headers, and footers
   - Whitespace analysis

## Project Structure

- `tables.py`: Functions for table detection and extraction
- `visualisation.py`: Functions for creating visualizations
- `header_footer.py`: Header and footer extraction functions
- `layout.py`: Layout analysis functions
- `seperator.py`: Column separator detection functions
- `pdf_layout_analyzer.py`: Main script containing the combined analysis

## Example

```python
# Analyze a single PDF
result = combined_pdf_layout_analysis("./samples/dual_col/6675433.pdf")

# Print summary
print(f"Overall classification: {result['overall_classification']}")
print(f"Single-column pages: {result['single_column_pages']}")
print(f"Dual-column pages: {result['dual_column_pages']}")
```

## Advanced Features

- **Whitespace Analysis**: Detects vertical whitespace to improve column separation
- **Feature-based Classification**: Uses multiple features for robust layout classification
- **Adaptive Thresholding**: Improves text detection in varying document qualities
- **Confidence-based Filtering**: Reduces noise in text detection

## License

[MIT License]

