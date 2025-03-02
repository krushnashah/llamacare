import json
from Bio import Entrez
from xml.etree import ElementTree as ET
import os
import time  # For rate-limiting

# Set up Entrez API details
Entrez.email = "kshah@ucsb.edu"  # Replace with your email

ARTICLES_DIR = "articles"  # Folder to store articles

# Function to get existing articles in the folder
def get_existing_pmcids(directory):
    """Retrieve existing PMCID filenames from the articles folder."""
    if not os.path.exists(directory):
        return set()
    return {filename.replace("article_", "").replace(".json", "") for filename in os.listdir(directory) if filename.endswith(".json")}

# Function to search for articles with free full text
def search_articles_with_full_text(topic, max_results=10):
    """
    Search for the top articles on a topic with free full text available on PubMed Central.
    """
    query = f'{topic} AND "open access"[Filter]'
    search_handle = Entrez.esearch(db="pmc", term=query, retmax=max_results, sort="relevance")
    search_results = Entrez.read(search_handle)
    search_handle.close()
    return set(search_results["IdList"])

# Function to fetch article summaries
def get_article_summaries(article_ids):
    """Fetch summaries (titles, PMCID, and links) for a list of article IDs."""
    summaries = []
    if not article_ids:
        return summaries

    print(f"Fetching summaries for {len(article_ids)} articles...")
    summary_handle = Entrez.esummary(db="pmc", id=",".join(article_ids), retmode="xml")
    summary_results = Entrez.read(summary_handle)
    summary_handle.close()

    for doc_summary in summary_results:
        pmcid = doc_summary.get("Id")  # PMCID
        title = doc_summary.get("Title")
        link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/" if pmcid else "No Link"
        summaries.append({"title": title, "pmcid": f"PMC{pmcid}", "link": link})
    return summaries

# Function to fetch full-text details using PMCID
def fetch_article_details_by_pmcid(pmcid):
    """Fetch full-text article details from PubMed Central using the PMCID."""
    print(f"Fetching full text for PMCID: {pmcid}")
    pmcid = pmcid.replace("PMC", "")
    time.sleep(0.5)  # 500ms delay between requests
    fetch_handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    article_data = fetch_handle.read()
    fetch_handle.close()
    return article_data

# Function to preprocess article data
def preprocess_article(article_data):
    """Preprocess article XML to extract metadata, full text, tables, and images."""
    root = ET.fromstring(article_data)

    title = root.findtext(".//article-title") or "No Title"
    journal = root.findtext(".//journal-title") or "No Journal"
    pub_date = root.findtext(".//pub-date/year") or "No Publication Date"

    abstract = " ".join(abstract.text for abstract in root.findall(".//abstract//p") if abstract.text) or "No Abstract"
    body = extract_full_text(root.find(".//body"))

    tables = extract_tables(root.findall(".//table-wrap"))
    images = extract_images(root.findall(".//fig"))

    if not body:
        return None  # Skip articles without meaningful body text

    return {
        "title": title.strip(),
        "journal": journal.strip(),
        "publication_date": pub_date.strip(),
        "abstract": abstract.strip(),
        "body": body.strip(),
        "tables": tables,
        "images": images,
    }

# Helper function to extract full text
def extract_full_text(element):
    if element is None:
        return ""
    texts = []
    if element.text:
        texts.append(element.text.strip())
    for child in element:
        texts.append(extract_full_text(child))
    if element.tail:
        texts.append(element.tail.strip())
    return " ".join(texts)

# Helper function to extract table data
def extract_tables(table_elements):
    tables = []
    for table in table_elements:
        title = table.findtext(".//title") or "No Title"
        content = extract_full_text(table.find(".//table"))
        tables.append({"title": title.strip(), "content": content.strip()})
    return tables

# Helper function to extract image data
def extract_images(fig_elements):
    images = []
    for fig in fig_elements:
        caption = extract_full_text(fig.find(".//caption")) or "No Caption"
        images.append({"caption": caption.strip()})
    return images

# Function to save article data as JSON
def save_article_to_file(article, pmcid, output_dir=ARTICLES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"article_{pmcid}.json")
    
    if os.path.exists(file_path):
        print(f"⚠️ Skipping existing article: {file_path}")
        return

    print(f"Saving article {pmcid} to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(article, f, indent=4)
    print(f"✅ Successfully saved: {file_path}")

# Main function to search, fetch, and save new articles
def main():
    topic = "polycystic ovary syndrome"
    max_results = 2500

    print(f"Checking for existing articles in '{ARTICLES_DIR}'...")
    existing_pmcids = get_existing_pmcids(ARTICLES_DIR)
    existing_pmcids = {pmcid.replace("PMC", "") for pmcid in existing_pmcids}  # Normalize format
    print(f"Found {len(existing_pmcids)} existing articles.")

    print(f"Searching for new articles on '{topic}' with free full text...")
    all_article_ids = search_articles_with_full_text(topic, max_results)
    all_article_ids = {pmcid.replace("PMC", "") for pmcid in all_article_ids}  # Normalize format
    
    new_article_ids = all_article_ids - existing_pmcids
    print(f"Existing PMCID filenames: {existing_pmcids}")
    print(f"Fetched PMCID from API: {all_article_ids}")
    print(f"New unique articles to fetch: {new_article_ids}")

    if not new_article_ids:
        print("⚠️ No unique new articles to fetch. Exiting.")
        return

    summaries = get_article_summaries(new_article_ids)

    for summary in summaries:
        print(f"Processing article: {summary['title']} (PMCID: {summary['pmcid']})")
        try:
            raw_data = fetch_article_details_by_pmcid(summary["pmcid"])
            print("Fetched article data successfully.")
            preprocessed_data = preprocess_article(raw_data)
            if preprocessed_data:
                save_article_to_file(preprocessed_data, summary["pmcid"])
            else:
                print(f"Skipping article with PMCID: {summary['pmcid']} (No meaningful content).")
        except Exception as e:
            print(f"Error processing article with PMCID: {summary['pmcid']}. Error: {e}")

    print(f"Total articles now in the folder: {len(get_existing_pmcids(ARTICLES_DIR))}")
    print("All new articles processed.")

if __name__ == "__main__":
    main()
