

def extract_header_footer_text(header_blocks, footer_blocks):
    """
    Extracts and formats header and footer text with coordinates.
    """
    header_data = {
        "text": " ".join([block['text'] for block in header_blocks]),
        "blocks": [{
            "text": block['text'],
            "coords": [float(block['left']), float(block['top']), 
                      float(block['left'] + block['width']), float(block['top'] + block['height'])]
        } for block in header_blocks]
    }
    
    footer_data = {
        "text": " ".join([block['text'] for block in footer_blocks]),
        "blocks": [{
            "text": block['text'],
            "coords": [float(block['left']), float(block['top']), 
                      float(block['left'] + block['width']), float(block['top'] + block['height'])]
        } for block in footer_blocks]
    }
    
    return header_data, footer_data