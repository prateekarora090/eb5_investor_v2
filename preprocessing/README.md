# Preprocessing Module and Usage Contract

## Directory Structure
```
preprocessing/
├── outputs/
│   ├── preprocessed_data/
│   └── preprocessing.log
├── README.md
├── document_preprocessor.py
└── __init__.py
...
tools/
├── google_drive_reader.py
...
secrets/
├── .env
```

NOTE: This uses the `google_drive_reader` tool, which uses `.env` file under `secrets/` for API keys.

## Overview
This module converts raw investment data (PDFs and websites) into structured, embeddable format for efficient analysis and retrieval.

## Output Structure
For each investment, the following files are generated in the `preprocessing/outputs/preprocessed_data/{investment_id}/` directory:

1. `metadata.json`: Overall investment information
2. For each PDF file:
   - `{filename}_chunks.json`: Contains text and visual chunks
   - `{filename}_text_embeddings.npy`: Text content embeddings
   - `{filename}_visual_embeddings.npy`: Visual content embeddings
3. For each website:
   - `{website_name}_chunks.json`: Contains text chunks
   - `{website_name}_embeddings.npy`: Text content embeddings

## Logging
Preprocessing progress and any errors are logged to `preprocessing/outputs/preprocessing.log`.

## Usage Contract

### 1. Accessing Metadata
- Read `metadata.json` to get an overview of processed files and websites.

### 2. Accessing Chunks
- Load `{filename}_chunks.json` or `{website_name}_chunks.json` to access raw text chunks.
- Use these for displaying context or for further processing.

### 3. Using Embeddings
- Load `*_embeddings.npy` files using `numpy.load()`.
- Use these embeddings for:
  a. Semantic search within documents
  b. Clustering similar content
  c. Input for machine learning models

### 4. Combining Text and Visual Content (for PDFs)
- Text embeddings (`{filename}_text_embeddings.npy`) represent the main content.
- Visual embeddings (`{filename}_visual_embeddings.npy`) represent content from images, charts, etc.
- Consider both when analyzing PDFs with significant visual elements.

### 5. Semantic Search Implementation
To perform semantic search:
1. Load the query embedding using the same embedding model.
2. Compute cosine similarity between the query embedding and document embeddings.
3. Retrieve the top-k most similar chunks.

### 6. Context Retrieval
When needing context for a specific part of a document:
1. Identify the relevant chunk using embeddings.
2. Retrieve the corresponding raw text from `*_chunks.json`.
3. Optionally, fetch neighboring chunks for more context.

### 7. Updating Preprocessed Data
- The preprocessing step is idempotent. Rerun on new or updated investments.
- Existing preprocessed data will be skipped unless manually deleted.

## Best Practices
1. Always refer to `metadata.json` first to understand the structure of preprocessed data.
2. Handle potential missing data (e.g., no visual embeddings for text-only PDFs).
3. Consider both text and visual embeddings when analyzing PDF content.
4. Use chunked data for displaying context to users or for fine-grained analysis.

## Example Usage (Python pseudo-code):

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load metadata
with open('preprocessed_data/investment_id/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load embeddings for a specific file
text_embeddings = np.load('preprocessed_data/investment_id/filename_text_embeddings.npy')
visual_embeddings = np.load('preprocessed_data/investment_id/filename_visual_embeddings.npy')

# Load chunks
with open('preprocessed_data/investment_id/filename_chunks.json', 'r') as f:
    chunks = json.load(f)

# Perform semantic search
query = "investment risks"
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode(query)

similarities = np.dot(text_embeddings, query_embedding)
top_k_indices = np.argsort(similarities)[-5:]  # Top 5 most similar chunks

for idx in top_k_indices:
    print(chunks['text_chunks'][idx])  # Display relevant text chunks
```

This documentation provides a clear contract for how other components in your system should interact with and utilize the preprocessed data.