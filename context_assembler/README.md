# Context Assembler

The `ContextAssembler` class is responsible for assembling and managing the context for EB-5 investment analysis. It provides functionality to load preprocessed data and perform semantic searches within the assembled context.

## Key Features

1. **Context Assembly**: Loads preprocessed data for a given investment ID, including metadata, document chunks, and website content.

2. **Investment Overview:**
   - Generates a concise overview of the investment, including summaries of each document and the determined investment sector.

3. **Semantic Search Tools:**
   - Provides two tools for semantic search:
     - `SearchAllDocumentsTool`:  Searches across all documents within the assembled investment context.
     - `SearchSpecificDocumentTool`: Searches within a specific document using its name.
   - These tools use sentence embeddings for accurate and relevant results.

## Usage

```python
from context_assembler import ContextAssembler

# Initialize the ContextAssembler
assembler = ContextAssembler('path/to/preprocessed_data')

# Assemble context for a specific investment
context = assembler.assemble_context('investment_id')

# Getting an investment overview
overview = assembler.get_investment_overview('investment_id')
print(overview)

# Using the Search tools
search_all_tool = SearchAllDocumentsTool(assembler)
search_results = search_all_tool.run(investment_id='investment_id', query='search query', top_k=5)

search_specific_tool = SearchSpecificDocumentTool(assembler)
specific_results = search_specific_tool.run(investment_id='investment_id', document_name='document_name', query='search_query')
```

## Methods

- `assemble_context(investment_id)`: Assembles the context for a given investment ID.
- `semantic_search(context, query, top_k=5)`: Performs a semantic search within the given context.
- `get_investment_overview(investment_id)`: Provides an overview of the investment, including document summaries and the investment sector.
- `semantic_search(context, query, top_k=5)`: (Internal method) Performs semantic search within a provided context.
- `summarize_document(chunks)`: (Internal method) Summarizes a document using its chunk embeddings.
- `determine_sector(overview)`: (Internal method) Determines the investment sector based on the provided overview text.

## Dependencies

- sentence_transformers
- numpy
- crewai_tools
- pydantic

Ensure these dependencies are installed before using the ContextAssembler.