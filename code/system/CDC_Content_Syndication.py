import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from fpdf import FPDF
import re
from docx import Document
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# Suppress FPDF deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='fpdf')

api_key = 'AIzaSyBKhFSMxFdXJ0KuHheIlcka9ct0g7icfdA'  
cse_id = '02e87f3b328f14ab7' 

# Folder containing internal documents
internal_documents_folder = 'data/CHLA_Converted_Documentation'  

# Folder where you want to save the PDFs
output_folder = 'CDC PDFS'  

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Calculate the date restrict parameter for the last 5 years
current_year = datetime.now().year
date_restrict = f"y[{current_year - 5}-01-01]"

# Function to perform Google Search using Custom Search JSON API
def google_search(query, api_key, cse_id, date_restrict, start=1):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}&dateRestrict={date_restrict}&num=10&start={start}"
    response = requests.get(url)
    return response.json()

# Function to compare text similarity using TF-IDF and cosine similarity
def compare_texts_tfidf(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

# Function to fetch the full text from a URL
def fetch_full_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        # Ensure the content is decoded properly
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraph text from the webpage
        paragraphs = soup.find_all('p')
        full_text = ' '.join([para.get_text() for para in paragraphs])
        return full_text
    except requests.RequestException as e:
        print(f"Failed to retrieve full text from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing content from {url}: {e}")
        print("Raw content:")
        print(response.text[:500])  # Print the first 500 characters for inspection
        return None

# Function to read content from a DOCX file
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to save text to a PDF file
def save_text_to_pdf(text, title, url, folder_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add DejaVuSans font without the deprecated uni parameter
    pdf.add_font("DejaVu", '', 'DejaVuSans.ttf')
    pdf.set_font("DejaVu", size=12)
    
    pdf.multi_cell(0, 10, text)

    # Add the URL at the end of the PDF
    pdf.add_page()
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(0, 10, f"Source URL: {url}")
    
    # Sanitize the title to create a valid filename
    filename = re.sub(r'[\\/*?:"<>|]', "", title) + ".pdf"
    filepath = os.path.join(folder_path, filename)
    pdf.output(filepath, 'F')
    return filepath


# Loop through all DOCX files in the folder
for filename in os.listdir(internal_documents_folder):
    if filename.endswith('.docx'):
        print("="*50)
        print(f"⭐ Processing document: {filename} ⭐")
        print("="*50)
        
        internal_document_path = os.path.join(internal_documents_folder, filename)
        
        # Read content from the internal document (DOCX)
        internal_document = read_docx(internal_document_path)
        document_title = os.path.splitext(filename)[0]
        
        # Remove the prefix "IC - ###.# " from the title
        cleaned_title = re.sub(r'^IC - \d+\.\d+ ', '', document_title)

        # Set the query to the cleaned title of the internal document
        query = cleaned_title + " site:cdc.gov"  # Adding site restriction to CDC

        results_data = []
        start_index = 1

        while len(results_data) < 10:
            # Retrieve CDC content
            results = google_search(query, api_key, cse_id, date_restrict, start=start_index)
            if 'items' not in results:
                print(f"No more results available for query: {query}")
                break

            # Compare internal document with CDC content and fetch full texts
            for item in results['items']:
                title = item['title']
                snippet = item['snippet']
                url = item['link']
                
                full_text = fetch_full_text(url)
                if full_text:
                    similarity = compare_texts_tfidf(internal_document, full_text)
                    results_data.append((similarity, title, url, full_text))
                    if len(results_data) >= 10:
                        break

            start_index += 10

        # Ensure we have exactly 10 results
        results_data.sort(reverse=True, key=lambda x: x[0])
        top_results = results_data[:10]

        # Save top 10 results into PDFs
        for idx, (similarity, title, url, full_text) in enumerate(top_results):
            filepath = save_text_to_pdf(full_text, title, url, output_folder)
            print(f"Saved '{title}' (Similarity: {similarity:.2f}) to {filepath}")
            print(f"URL: {url}\n")

print("Processing Completed")
