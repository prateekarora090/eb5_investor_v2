import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from crewai_tools import BaseTool
from typing import Type, Any, ForwardRef
from pydantic.v1 import BaseModel, Field, create_model, ConfigDict
from sklearn.metrics.pairwise import cosine_similarity

class ContextAssembler(BaseModel):
    model_config = ConfigDict(from_attributes=True) # Allows 
    preprocessed_data_dir: str = Field(..., description="preprocessing directory, to assemble the context")
    model: Any = Field(..., description="model used to assemble the context")
    # class Config:
        # arbitrary_types_allowed = True

    def __init__(self, preprocessed_data_dir):
        super().__init__(preprocessed_data_dir=preprocessed_data_dir, model=SentenceTransformer('all-MiniLM-L6-v2'))
        self.preprocessed_data_dir = preprocessed_data_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    # @classmethod
    # def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> dict[str, Any]:
    #     """Defines the Pydantic schema for ContextAssembler."""
    #     return handler(create_model('ContextAssemblerModel', preprocessed_data_dir=(str, ...)))
    
    def assemble_context(self, investment_id):
        """ Most important function that compiles all documents and websites
        per option to return a dictionary of all "context" for that option."""

        investment_dir = os.path.join(self.preprocessed_data_dir, investment_id)
        
        # Load metadata
        with open(os.path.join(investment_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        context = {
            'metadata': metadata,
            'documents': [],
            'websites': []
        }

        # Load document chunks
        for file_name in metadata['folder_files']:
            chunks_file = os.path.join(investment_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    doc_chunks = json.load(f)
                context['documents'].append(doc_chunks)

        # Load website chunks
        for website in metadata['websites']:
            website_file = website.replace('https://', '').replace('http://', '').replace('/', '_')
            chunks_file = os.path.join(investment_dir, f"{website_file}_chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    website_chunks = json.load(f)
                context['websites'].append(website_chunks)

        return context

    def semantic_search(self, context, query, top_k=5):
        """Searches the context assembled by assemble_context() by comparing
        the embedding of the query against the embeddings of each document. """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        results = []
        for doc in context['documents']:
            for chunk in doc['text_chunks']:
                chunk_embedding = self.model.encode(chunk, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding)
                results.append((similarity.item(), chunk, doc['name']))
        
        for website in context['websites']:
            for chunk in website['chunks']:
                chunk_embedding = self.model.encode(chunk, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding)
                results.append((similarity.item(), chunk, website['url']))
        
        results.sort(reverse=True, key=lambda x: x[0])
        return results[:top_k]
    
    def get_investment_overview(self, investment_id):
        """Provides a broad overview of the investment, including document
        descriptions. Helpful to provide to agents early on in the workflow."""
        
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
            chunks_file = os.path.join(investment_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    doc_chunks = json.load(f)
                summary = self.summarize_document(doc_chunks['text_chunks'])
                overview += f"- **{file_name}:** {summary}\n"

        for website in metadata['websites']:
            website_file = website.replace('https://', '').replace('http://', '').replace('/', '_')
            chunks_file = os.path.join(investment_dir, f"{website_file}_chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    website_chunks = json.load(f)
                summary = self.summarize_document(website_chunks['chunks'])
                overview += f"- **{website}:** {summary}\n"

        investment_sector = self.determine_sector(overview)
        overview += f"\n**Investment Sector:** {investment_sector}"

        return overview

    # def summarize_document(self, chunks):
    #     """Summarizes a document by averaging its chunk embeddings."""
    #     embeddings = [self.model.encode(chunk) for chunk in chunks]
    #     avg_embedding = np.mean(embeddings, axis=0)
    #     summary = self.model.decode(avg_embedding)
    #     return summary

    def summarize_document(self, chunks, max_summary_length=200):
        """Summarizes a document by selecting the most representative chunks."""
        if not chunks:
            return "No content available for summarization."

        # Encode all chunks
        embeddings = self.model.encode(chunks)

        # Calculate the centroid (mean) of all embeddings
        centroid = np.mean(embeddings, axis=0)

        # Calculate cosine similarity between each chunk and the centroid
        similarities = cosine_similarity([centroid], embeddings)[0]

        # Sort chunks by similarity to the centroid
        sorted_chunks = [chunk for _, chunk in sorted(zip(similarities, chunks), reverse=True)]

        # Select top chunks until we reach the max summary length
        summary = ""
        for chunk in sorted_chunks:
            if len(summary) + len(chunk) <= max_summary_length:
                summary += chunk + " "
            else:
                break

        return summary.strip()

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
    investment_id: str = Field(..., description="ID of the investment to search within.")
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

        investment_context = self.context_assembler.assemble_context(investment_id)
        return self.context_assembler._semantic_search(investment_context, query, top_k)


### Exposed Tool #2: Searching a specific investment documetn!
class SearchSpecificDocumentSchema(BaseModel):
    """Input for SearchSpecificDocumentTool."""
    investment_id: str = Field(..., description="ID of the investment to search within.")
    document_name: str = Field(..., description="Name of the document to search.")
    query: str = Field(..., description="Search query.")
    top_k: int = Field(5, description="Number of top results to return.")

class SearchSpecificDocumentTool(BaseTool):
    name: str = "Search Specific Document"
    description: str = "Performs a semantic search within a specific document."
    args_schema: Type[BaseModel] = SearchSpecificDocumentSchema
    context_assembler: ContextAssembler = Field(..., description="context assembler", init_var=True)

    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types

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