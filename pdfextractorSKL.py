from PyPDF2 import PdfReader
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: PDF Parsing
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Step 2: Table of Contents Extraction
def extract_table_of_contents(text):
    # Regex pattern for identifying typical table of contents structures
    toc_pattern = r'(table\s*of\s*contents|contents)([\s\S]*?)(chapter\s*\d+|page\s*\d+)'
    toc_matches = re.findall(toc_pattern, text, re.IGNORECASE)
    #print(toc_matches)
    # Assuming TOC format is simple here, adjust regex as necessary for complex cases
    if toc_matches:
        toc_content = toc_matches[0][1]
        # Extracting chapter titles assuming chapters are labeled as "Chapter <number>"
        chapters = re.findall(r'(chapter\s*\d+)', toc_content, re.IGNORECASE)
        return chapters
    else:
        return None

# Step 3: Indexing
def index_book(text, chapters):
    index = {}
    start_indices = [text.index(chapter) for chapter in chapters]
    for i in range(len(chapters)):
        chapter_title = chapters[i]
        start_index = start_indices[i]
        end_index = start_indices[i + 1] if i < len(chapters) - 1 else len(text)
        chapter_content = text[start_index:end_index]
        index[chapter_title] = chapter_content
    return index

# Step 4: Text Vectorization
def vectorize_text(texts):
    tfidf_vectorizer = TfidfVectorizer()
    vectors = tfidf_vectorizer.fit_transform(texts)
    return vectors

# Example usage:
pdf_path = 'C:\python_dev\web\Higher Engineering Mathematics.pdf'
text = extract_text_from_pdf(pdf_path)
#print(text)
chapters = extract_table_of_contents(text)
#print(chapters)
if chapters:
    book_index = index_book(text, chapters)
    chapter_texts = list(book_index.values())
    vectorized_chapters = vectorize_text(chapter_texts)
    # Now you have indexed chapters and their vector representations
    print("Book Index:", book_index)
    print("Vectorized Chapters:", vectorized_chapters)
else:
    print("No table of contents found.")
