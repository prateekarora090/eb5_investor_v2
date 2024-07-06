import os
import json
import logging
from tools.google_drive_reader import list_files_in_folder, read_file_from_drive
from tools.web_scraper import scrape_website
from tools.pdf_reader import read_pdf
from sentence_transformers import SentenceTransformer
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DocumentPreprocessor:
    def __init__(self, base_dir='preprocessing/outputs', chunk_size=1000):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, 'preprocessed_data')
        self.log_file = os.path.join(base_dir, 'preprocessing.log')
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.total_files = 0
        self.processed_files = 0

    def preprocess_investments(self, investments_file):
        with open(investments_file, 'r') as f:
            investments = json.load(f)
        
        for i, investment in enumerate(investments, 1):
            self.logger.info(f"Processing investment {i}/{len(investments)}: {investment['name']}")
            self.preprocess_investment(investment)

    def preprocess_investment(self, investment):
        investment_dir = os.path.join(self.output_dir, investment['id'])
        os.makedirs(investment_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(investment_dir, 'metadata.json')):
            self.logger.info(f"Skipping already processed investment: {investment['name']}")
            return

        self.logger.info(f"Processing investment: {investment['name']}")
        
        folder_content = self.process_folder(investment['folder_id'], investment_dir)
        website_content = self.process_websites(investment['websites'], investment_dir)

        metadata = {
            'id': investment['id'],
            'name': investment['name'],
            'folder_files': [f['name'] for f in folder_content if f and 'name' in f],
            'websites': investment['websites']
        }

        with open(os.path.join(investment_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Preprocessed investment {investment['name']} saved to {investment_dir}")

    def process_folder(self, folder_id, investment_dir):
        self.logger.info(f"Processing folder: {folder_id}")
        files = list_files_in_folder(folder_id)
        self.total_files += len(files)
        self.logger.info(f"Found {len(files)} files in folder. Total files: {self.total_files}")

        results = []
        for file in files:
            result = self.process_file(file, investment_dir)
            if result:
                results.append(result)
        
        if not results:
            self.logger.warning(f"No files were successfully processed in folder: {folder_id}")
        
        return results

    def process_file(self, file, investment_dir):
        self.processed_files += 1
        self.logger.info(f"Processing file {self.processed_files}/{self.total_files}: {file['name']} (Type: {file['mimeType']})")
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            if not file['name'].startswith("(ignored) "):
                return self.process_folder(file['id'], investment_dir)
        elif file['mimeType'] == 'application/pdf':
            try:
                self.logger.info(f"Reading PDF file {self.processed_files}/{self.total_files}: {file['name']}")
                file_content = read_file_from_drive(file['id'])
                pdf_content = read_pdf(file_content)
                
                text_chunks = self.chunk_text(pdf_content['text_content'])
                visual_chunks = self.chunk_text(pdf_content['visual_content'])
                
                if not text_chunks and not visual_chunks:
                    self.logger.warning(f"No content extracted from file: {file['name']}")
                    return None

                file_data = {
                    "name": file['name'],
                    "text_chunks": text_chunks,
                    "visual_chunks": visual_chunks,
                    "text_chunk_count": len(text_chunks),
                    "visual_chunk_count": len(visual_chunks)
                }

                if text_chunks:
                    text_embeddings = self.embed_chunks(text_chunks)
                    np.save(os.path.join(investment_dir, f"{os.path.splitext(file['name'])[0]}_text_embeddings.npy"), text_embeddings)
                
                if visual_chunks:
                    visual_embeddings = self.embed_chunks(visual_chunks)
                    np.save(os.path.join(investment_dir, f"{os.path.splitext(file['name'])[0]}_visual_embeddings.npy"), visual_embeddings)
                
                file_base_name = os.path.splitext(file['name'])[0]
                with open(os.path.join(investment_dir, f"{file_base_name}_chunks.json"), 'w') as f:
                    json.dump(file_data, f)
                
                self.logger.info(f"File {self.processed_files}/{self.total_files}: {file['name']} processed and saved successfully")
                return file_data
            except Exception as e:
                self.logger.error(f"Error processing file {self.processed_files}/{self.total_files}: {file['name']}: {str(e)}", exc_info=True)
                return {
                    "name": file['name'],
                    "error": str(e)
                }
        return None

    def process_websites(self, websites, investment_dir):
        website_content = []
        for website in websites:
            try:
                self.logger.info(f"Scraping website: {website}")
                content = scrape_website(website)
                self.logger.info(f"Website content scraped. Size: {len(content)}")
                
                chunks = self.chunk_text(content)
                self.logger.info(f"Website content chunked. Number of chunks: {len(chunks)}")
                
                embeddings = self.embed_chunks(chunks)
                self.logger.info(f"Website embeddings created. Shape: {embeddings.shape}")
                
                website_data = {
                    "url": website,
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                }
                
                website_file_name = website.replace('https://', '').replace('http://', '').replace('/', '_')
                with open(os.path.join(investment_dir, f"{website_file_name}_chunks.json"), 'w') as f:
                    json.dump(website_data, f)
                
                np.save(os.path.join(investment_dir, f"{website_file_name}_embeddings.npy"), embeddings)
                
                website_content.append(website_data)
                self.logger.info(f"Website {website} processed and saved successfully")
            except Exception as e:
                self.logger.error(f"Error scraping website {website}: {str(e)}", exc_info=True)
        return website_content

    def chunk_text(self, text):
        words = text.split()
        return [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def embed_chunks(self, chunks):
        if not chunks:
            return np.array([])  # Return an empty numpy array if chunks is empty
        return self.embedding_model.encode(chunks)