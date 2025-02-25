# PDF Medical Health Records Processing

## Overview
This repository contains code for processing PDF medical health records and converting them into readable markup text in JSON format. The project leverages OpenCV, PDF libraries in Python, and AI-based models like Vision Transformers (ViT) and Layout Parser to perform Document Layout Analysis (DLA). The processed output ensures structured and accessible medical records for further analysis.

## Features
- **Document Layout Analysis (DLA):** Extracts and processes structured text from medical PDFs.
- **Footer & Header Removal:** Accurately detects and removes repetitive footers and headers across multiple pages.
- **JSON Conversion:** Outputs structured, readable JSON format for easy parsing and further usage.
- **AI-Based Processing:** Implements ViT and Layout Parser for enhanced document segmentation.

## Installation
To use this repository, clone it and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/purrate/AGS_intern.git
cd [Directory name]

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the main processing script to convert PDFs into JSON format:

```bash
python process_pdf.py --input path/to/medical_record.pdf --output path/to/output.json
```

### Arguments:
- `--input`: Path to the input PDF file.
- `--output`: Path to save the processed JSON output.

## Output Format
The extracted text is structured into a JSON file, with sections categorized based on document layout analysis.

```json
{
  "patient_info": {
    "name": "John Doe",
    "age": 45,
    "gender": "Male"
  },
  "medical_history": [
    {
      "date": "2024-02-10",
      "diagnosis": "Hypertension",
      "treatment": "Prescribed medication X"
    }
  ],
  "footer_removed": true
}
```

## Project Structure
```
AGS_Health/
│-- process_pdf.py         # Main script to process PDFs
│-- utils.py               # Utility functions for PDF processing
│-- models/                # AI-based models for document layout analysis
│-- samples/               # Sample medical PDFs for testing
│-- output/                # Processed JSON files
│-- requirements.txt       # List of dependencies
```

## Future Enhancements
- Implementing OCR for handwritten medical notes.
- Enhancing entity recognition using NLP models.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

