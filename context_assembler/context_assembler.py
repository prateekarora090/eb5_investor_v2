import os
import json
import logging
import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
from crewai_tools import BaseTool
from typing import Type, Any, ForwardRef
from pydantic.v1 import BaseModel, Field, create_model, ConfigDict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import tiktoken

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='eb5_analysis.log'
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv('secrets/.env')

# Set API keys
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_AI_API_KEY')
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

def get_llm(model_name="gemini-pro"):
    if model_name == "gemini-pro":
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    elif model_name == "gpt-3.5-turbo":
        return ChatOpenAI(model_name="gpt-3.5-turbo")
    # Add more model options as needed

class ContextAssembler(BaseModel):
    """
    Assembles and processes context for a given investment ID, 
    providing various methods to search, summarize, and determine sector information.
    """
    model_config = ConfigDict(from_attributes=True) # Allows 
    preprocessed_data_dir: str = Field(..., description="preprocessing directory, to assemble the context")
    model: Any = Field(..., description="model used to assemble the context")
    llm: Any = Field(..., description="internal LLM used to summarize documents to assemble context")
    # class Config:
        # arbitrary_types_allowed = True

    def __init__(self, preprocessed_data_dir):
        super().__init__(preprocessed_data_dir=preprocessed_data_dir, model=SentenceTransformer('all-MiniLM-L6-v2'), llm=get_llm())
        self.preprocessed_data_dir = preprocessed_data_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = get_llm()
    
    def assemble_context(self, investment_id, include_full_chunks=False):
        """Compiles all documents and websites per option to return a dictionary of all "context"
           for that option. Note: This function is not very summary-like.

        Args:
            investment_id (str): The ID of the investment.
            include_full_chunks (bool, optional): Whether to include the full chunks in the context. Defaults to False.

        Returns:
            dict: A dictionary containing the metadata, documents, and websites for the given investment.
                Here's the sample structure:    
                    {
                        'metadata': dict,
                        'documents': list[dict],
                        'websites': list[dict]
                    }
                Further, each document and website is a dictionary with the following keys:
                    'documents':
                        {
                            'file': str, # Name of the file
                            'summary': str, # Summary of the document content
                            'chunks': list[str] # List of chunks, if include_full_chunks is True
                        }
                    'websites':
                        {
                            'url': str, # URL of the website
                            'summary': str, # Summary of the website content
                            'chunks': list[str] # List of chunks, if include_full_chunks is True
                        }
        """
        investment_dir = os.path.join(self.preprocessed_data_dir, investment_id)
        # DEBUG: logger.info(f"assemble_context() called on {investment_id}.")
        
        # Load metadata
        with open(os.path.join(investment_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        context = {
            'metadata': metadata,
            'documents': [],
            'websites': []
        }

        # For files:
        # chunks_file = os.path.join(investment_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
        # For websites:
        # chunks_file = os.path.join(investment_dir, f"{website_file}_chunks.json")

        # 1) Load file chunks and get summaries (and generate from chunks, if needed)
        for file_name in metadata['folder_files']:
            summary = self.get_or_create_summary(investment_dir, file_name)
            doc_info = {
                'file': file_name,
                'summary': summary
            }

            # Still return the full chunks, if requested
            if include_full_chunks:
                chunks_file = os.path.join(investment_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
                if os.path.exists(chunks_file):
                    with open(chunks_file, 'r') as f:
                        doc_info['chunks'] = json.load(f)['text_chunks']
            context['documents'].append(doc_info)

        # 2) Load website chunks and get summaries (and generate from chunks, if needed)
        for website in metadata['websites']:
            website_file = website.replace('https://', '').replace('http://', '').replace('/', '_')
            summary = self.get_or_create_summary(investment_dir, website_file, is_website=True)
            website_info = {
                'url': website,
                'summary': summary
            }

            # Still return the full chunks, if requested
            if include_full_chunks:
                chunks_file = os.path.join(investment_dir, f"{website_file}_chunks.json")
                if os.path.exists(chunks_file):
                    with open(chunks_file, 'r') as f:
                        website_info['chunks'] = json.load(f)['chunks']
            context['websites'].append(website_info)

        logging.info(f"Results for assemble_context() on {investment_id}: {context}")
        return context

        # DEBUG: logger.info(f"Results for assemble_context() on {investment_id} look like: {context}")
        return context
    
    def get_or_create_summary(self, investment_dir, file_name, is_website=False):
        """Returns summary of a document or website.
        
        This method retrieves the summary of a document or website. If the summary
        is already cached, it is returned directly. Otherwise, the method generates
        the summary using an LLM (Language Model) and caches it for future use.
        
        Args:
            investment_dir (str): The directory where the investment files are stored.
            file_name (str): The name of the file or website.
            is_website (bool, optional): Indicates whether the input is a website or not.
                Defaults to False.
        
        Returns:
            str: The summary of the document or website.
        
        Raises:
            FileNotFoundError: If the summary file or chunks file does not exist.
        """
        summary_file = os.path.join(investment_dir, f"{file_name}_summary.txt")
        if is_website:
            chunks_file = os.path.join(investment_dir, f"{file_name}_chunks.json")
        else:
            chunks_file = os.path.join(investment_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
        
        # If summary exists for the file or website, return it.
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return f.read()
        
        # If not, 1) get the chunks
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            
            # 2) combine and summarize chunks
            if is_website:
                summary = self.summarize_existing_chunks(chunks['chunks'], file_name)
            else:
                summary = self.summarize_existing_chunks(chunks['text_chunks'], file_name)
                
            # 3) store the result!
            with open(summary_file, 'w') as f:
               f.write(summary)
            
            return summary

        return "No content available for summarization."

    def summarize_existing_chunks(self, chunks, file_name):
        """Summarize the given chunks using an LLM (Language Model).

        Args:
            chunks (list): A list of chunks to be summarized.
            file_name (str): The name of the file being summarized.

        Returns:
            str: The final summary of the chunks.

        Raises:
            None

        Notes:
            This method uses the BART (Bidirectional and Auto-Regressive Transformer) model for summarization.
            It first summarizes each individual chunk and then combines the summaries into a single summary.

        """
        # [Research] Compared to other summarization models (pegasus, allenai), BART worked best!
        summaries = []
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

        for chunk in tqdm(chunks, desc="Processing chunks"):
            summary = summarizer(chunk, max_length=600, min_length=200)[0]['summary_text']
            summaries.append(summary)
        
        chunk_summaries = " ".join(summaries)
        final_summary = summarizer(chunk_summaries, max_length=3000, min_length=200)[0]['summary_text']
        return final_summary

    # def summarize_with_gemini(self, text):
    #     time.sleep(1)  # Add a 1-second delay between API calls
    #     prompt = PromptTemplate(
    #         input_variables=["text"],
    #         template="Please provide a concise summary of the following text in about 200 words:\n\n{text}"
    #     )
    #     chain = LLMChain(llm=self.llm, prompt=prompt)
    #     response = chain.run(text=text)
    #     return response.strip()

    def semantic_search(self, context, query, top_k=5):
        """Searches the context assembled by assemble_context() by comparing
        the embedding of the query against the embeddings of each document.

        Args:
            context (dict): The context assembled by the `assemble_context()` method.
                Ensure that this context contains 'chunks' for both documents and websites.
            query (str): The query string to search for.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            list: A list of tuples containing the search results. Each tuple contains the file or URL,
            the similarity score, and the corresponding chunk of text.

        Raises:
            KeyError: If the required keys 'documents' or 'websites' are not found in the context.

        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # DEBUG: logger.info(f"Semantic Search for for {query} on {context}")
        
        results = []
        # Search over documents
        for doc in context.get('documents', []):
            if (not doc.get('chunks', [])):
                logging.error(f"Need chunks for semantic_search(), not found in context: {context}")
            for chunk in doc.get('chunks', []):
                chunk_embedding = self.model.encode(chunk, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding)
                results.append((doc['file'], similarity.item(), chunk))
        
        # Search over websites (if they exist)
        for website in context.get('websites', []):
            if (not doc.get('websites', [])):
                logging.error(f"Need chunks for semantic_search(), not found in context: {context}")
            for chunk in website.get('chunks', []):
                chunk_embedding = self.model.encode(chunk, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding)
                results.append((website['url'], similarity.item(), chunk))
        
        results.sort(reverse=True, key=lambda x: x[0])
        # DEBUG: logger.info(f"Results for {query} on {context}: {results[:top_k]}")
        return results[:top_k]
    
    def search_specific_document(self, context, document_name, query, top_k=5):
        """
        Searches within a specific document.

        Args:
            context (dict): The context containing documents and websites.
            document_name (str): The name of the document to search within.
            query (str): The search query.
            top_k (int, optional): The number of top search results to return. Defaults to 5.

        Returns:
            list: A list of search results.

        """
        # Extract just the filename without the extension for PDF files
        if document_name.endswith(".pdf"):
            document_name = os.path.splitext(document_name)[0]

        # Search over documents
        for doc in context['documents']:
            if doc['file'].startswith(document_name):
                print("Found document!!!")
                return self.semantic_search({'documents': [doc]}, query, top_k)

        # Search over websites
        for website in context['websites']:
            if website['url'] == document_name:
                print("Found website!!!")
                return self.semantic_search({'websites': [website]}, query, top_k)

        print(f"ERROR: Could not find a document with inputted name {document_name}")
        return []  # Return empty list if document not found
    
    def get_investment_overview(self, investment_id):
        """Provides a broad overview of the investment, including document
        descriptions. Helpful to provide to agents early on in the workflow."""
        
        print(f"[context_assembler] Getting overview for {investment_id}...") 
        investment_dir = os.path.join(self.preprocessed_data_dir, investment_id)

        # Load metadata
        with open(os.path.join(investment_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Start an "overview"
        overview = f"**Investment ID:** {metadata['id']}\n\n"
        overview = f"**Investment Name:** {metadata['name']}\n\n"
        overview += "**Document Summaries:**"
        overview += f"""_This includes names of files and websites along with their summaries. 
            These can be searched using SearchAllDocuments or SearchSpecificDocument tools_\n"""

        for file_name in metadata['folder_files']:
            summary = self.get_or_create_summary(investment_dir, file_name, is_website=False)
            overview += f"- **{file_name}:** {summary}\n\n"

        for website in metadata['websites']:
            website_file = website.replace('https://', '').replace('http://', '').replace('/', '_')
            summary = self.get_or_create_summary(investment_dir, website_file, is_website=True)
            overview += f"- **{website}:** {summary}\n\n"

        investment_sector = self.determine_sector(overview)
        overview += f"\n**Investment Sector:** {investment_sector}"

        logger.info(f"Overview for {investment_id} is {overview}")
        return overview

    def determine_sector(self, overview):
        """Determines the investment sector based on keywords and phrases."""
        # TODO: Very, very basic implementation! Can improve this significantly.
        # TODO: Consider using keyword extraction, classification models
        # or external APIs for better accuracy.
        overview_lower = overview.lower()
        if "real estate" in overview_lower:
            if any(keyword in overview_lower for keyword in ["residential", "apartment", "housing"]):
                return "Residential Real Estate"
            elif any(keyword in overview_lower for keyword in ["commercial", "office", "retail", "industrial"]):
                return "Commercial Real Estate"
            else:
                return "Real Estate" # General real estate if specific type not found
        elif any(keyword in overview_lower for keyword in ["technology", "software", "saas", "ai"]):
            return "Technology"
        elif "healthcare" in overview_lower:
            return "Healthcare"
        # Add more sectors and keywords as needed
        else:
            return "Unknown"

### Exposed Tool #1: Searching across all investment documents!
class SearchAllDocumentsSchema(BaseModel):
    """Input for SearchAllDocumentsTool."""
    investment_id: str = Field(..., description="ID of the investment to search within. NOTE = This is NOT the investment name, it's the ID!")
    query: str = Field(..., description="Search query.")
    top_k: int = Field(5, description="Number of top results to return.")

class SearchAllDocumentsTool(BaseTool):
    name: str = "Search All Documents"
    description: str = "Performs a semantic search across all documents in the investment context."
    args_schema: Type[BaseModel] = SearchAllDocumentsSchema
    context_assembler: ContextAssembler = Field(..., description="context assembler", init_var=True)

    def __init__(self, context_assembler):
        super().__init__()
        self.context_assembler = context_assembler

    def _run(self, **kwargs: Any) -> Any:
        investment_id = kwargs.get("investment_id")
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", 5)  # Default to 5 if not specified

        # print(f"~~~~ [Tool Use] SearchAllDocs for {investment_id}: {query} and {top_k} ~~~~")
        investment_context = self.context_assembler.assemble_context(
            investment_id,
            include_full_chunks=True # TODO: Not sure if we really need this / even support this mode for assemble_context()
        )
        result = self.context_assembler.semantic_search(investment_context, query, top_k)
        # print(f"~~~~ [Tool Use] Output for SearchAllDocs for {investment_id}: {query} and {top_k} ~~~~")
        # print(f"~~~~ [Tool Use] Results: {result} ~~~~")
        return result


### Exposed Tool #2: Searching a specific investment documetn!
class SearchSpecificDocumentSchema(BaseModel):
    """Input for SearchSpecificDocumentTool."""
    investment_id: str = Field(..., description="ID of the investment to search within.  NOTE = This is NOT the investment name, it's the ID!")
    document_name: str = Field(..., description="Name of the document to search.")
    query: str = Field(..., description="Search query.")
    top_k: int = Field(5, description="Number of top results to return.")

class SearchSpecificDocumentTool(BaseTool):
    name: str = "Search Specific Document"
    description: str = "Performs a semantic search within a specific document."
    args_schema: Type[BaseModel] = SearchSpecificDocumentSchema
    context_assembler: ContextAssembler = Field(..., description="context assembler", init_var=True)

    def __init__(self, context_assembler):
        super().__init__()
        self.context_assembler = context_assembler

    def _run(self, **kwargs: Any) -> Any:
        investment_id = kwargs.get("investment_id")
        document_name = kwargs.get("document_name")
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", 5)

        investment_context = self.context_assembler.assemble_context(investment_id)
        return self.context_assembler.search_specific_document(investment_context, document_name, query, top_k)


# Usage example
if __name__ == "__main__":
    assembler = ContextAssembler('preprocessing/outputs/preprocessed_data')
    investment_id = '1' # Example investment ID

    # 1. Example of assembling context
    context = assembler.assemble_context(investment_id)
    print(f"Assembled context for investment_1:")
    print(f"Metadata: {context['metadata']}")
    print(f"Number of documents: {len(context['documents'])}")
    print(f"Number of websites: {len(context['websites'])}")

    # 2. Example of get_investment_overview()
    overview = assembler.get_investment_overview(investment_id)
    print(f"\nInvestment Overview (ID: {investment_id}):\n{overview}")

    # 3. Example of SearchAllDocumentsTool
    search_all_tool = SearchAllDocumentsTool(assembler)
    query = "job creation requirements"
    search_all_results = search_all_tool.run(investment_id=investment_id, query=query, top_k=3) 
    print(f"\nSearch All Documents Results (Query: '{query}'):\n")
    for similarity, chunk, source, chunk_index in search_all_results:
        print(f"- Similarity: {similarity:.4f}, Source: {source}, Chunk Index: {chunk_index}, Text: {chunk[:100]}...")

    # 4. Example of SearchSpecificDocumentTool
    search_specific_tool = SearchSpecificDocumentTool(assembler)
    document_name = "Confidential Offering Memorandum.pdf" 
    query = "use of funds"
    search_specific_results = search_specific_tool.run(investment_id=investment_id, document_name=document_name, query=query)
    print(f"\nSearch Specific Document Results (Document: '{document_name}', Query: '{query}'):\n")
    for similarity, chunk, source, chunk_index in search_specific_results:
        print(f"- Similarity: {similarity:.4f}, Source: {source}, Chunk Index: {chunk_index}, Text: {chunk[:100]}...")
    
    # Example semantic search
    query = "EB-5 visa requirements"
    search_results = assembler.semantic_search(context, query)
    print(f"\nTop 5 results for query '{query}':")
    for similarity, chunk, source in search_results:
        print(f"Similarity: {similarity:.4f}")
        print(f"Source: {source}")
        print(f"Chunk: {chunk[:100]}...")  # Print first 100 characters of the chunk
        print()