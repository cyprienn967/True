import os
from metapub import FindIt

#os.environ["api_key"] = "b6659448725729569fc1c036121f7de90507"
def extract_pmids(file_path):
    """Extract PMIDs from a file containing PubMed links."""
    pmids = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'pubmed/' in line:
                pmid = line.strip().split('/')[-1]
                pmids.append(pmid)
    return pmids

def fetch_pdf_urls(pmids, output_file):
    """Fetch PDF URLs for a list of PMIDs and save them to a file."""
    pdf_urls = {}
    with open(output_file, 'w') as f:
        for pmid in pmids:
            try:
                src = FindIt(pmid)
                if src.url:
                    pdf_urls[pmid] = src.url
                    f.write(f"{pmid}\t{src.url}\n")
                    print(f"PMID {pmid} PDF URL: {src.url}")
                else:
                    print(f"PMID {pmid}: PDF not found. Reason: {src.reason}")
            except Exception as e:
                print(f"An error occurred for PMID {pmid}: {e}")
    return pdf_urls

# Example usage
if __name__ == "__main__":
    input_file = "documents.txt"        # File containing PubMed links
    output_file = "pdf_urls.txt"        # Output file to store PDF URLs
    pmids = extract_pmids(input_file)
    fetch_pdf_urls(pmids, output_file)
