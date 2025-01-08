import requests
import os
from pathlib import Path

def read_pdf_urls(file_path):
    """Read PDF URLs from a file."""
    pdf_urls = {}
    with open(file_path, 'r') as f:
        for line in f:
            pmid, url = line.strip().split('\t')
            pdf_urls[pmid] = url
    return pdf_urls

def download_pdfs(pdf_urls, output_dir):
    """Download PDFs from a dictionary of PMIDs and URLs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for pmid, url in pdf_urls.items():
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            pdf_path = Path(output_dir) / f"{pmid}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded PDF for PMID {pmid}: {pdf_path}")
        except Exception as e:
            print(f"Failed to download PDF for PMID {pmid}: {e}")

if __name__ == "__main__":
    # Input file and output directory, relative to the current working directory
    input_file = "pdf_urls.txt"  # File in the current directory
    output_dir = "../pdf_docs"  # Navigate up one level to reach pdf_docs

    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read URLs and download PDFs
    pdf_urls = read_pdf_urls(input_file)
    download_pdfs(pdf_urls, output_dir)
