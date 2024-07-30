import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import torch

# Add the parent directory to the Python path to allow importing from context_assembler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the SentenceTransformer and get_llm imports
mock_sentence_transformer = MagicMock()
mock_get_llm = MagicMock()

# Apply the mocks
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers'].SentenceTransformer = mock_sentence_transformer
sys.modules['sentence_transformers'].util = MagicMock()
sys.modules['context_assembler.context_assembler'].SentenceTransformer = mock_sentence_transformer
sys.modules['context_assembler.context_assembler'].get_llm = mock_get_llm

from context_assembler.context_assembler import ContextAssembler, SearchAllDocumentsTool, SearchSpecificDocumentTool

class TestContextAssembler(unittest.TestCase):
    def setUp(self):
        self.assembler = ContextAssembler('test_preprocessed_data')

    @patch('context_assembler.context_assembler.os.path.exists')
    @patch('context_assembler.context_assembler.open')
    @patch('context_assembler.context_assembler.json.load')
    def test_assemble_context(self, mock_json_load, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_json_load.side_effect = [
            {"id": "1", "name": "Test Investment", "folder_files": ["doc1.pdf"], "websites": ["https://example.com"]},
            {"text_chunks": ["Chunk 1", "Chunk 2"]},
            {"chunks": ["Chunk 1", "Chunk 2"]}
        ]
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            'Summary of doc1',
            'Summary of website'
        ]

        context = self.assembler.assemble_context('1', include_full_chunks=True)

        self.assertIn('metadata', context)
        self.assertIn('documents', context)
        self.assertIn('websites', context)
        self.assertEqual(len(context['documents']), 1)
        self.assertEqual(len(context['websites']), 1)
        self.assertEqual(context['documents'][0]['file'], 'doc1.pdf')
        self.assertEqual(context['websites'][0]['url'], 'https://example.com')
        self.assertIn('chunks', context['documents'][0])
        self.assertIn('chunks', context['websites'][0])

    @patch('context_assembler.context_assembler.util.pytorch_cos_sim')
    def test_semantic_search(self, mock_cos_sim):
        mock_context = {
            'documents': [{'file': 'doc1.pdf', 'chunks': ['This is a test chunk']}],
            'websites': [{'url': 'https://example.com', 'chunks': ['This is another test chunk']}]
        }

        # Mock the encode method to return a tensor of the correct type
        self.assembler.model.encode.return_value = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        # Mock the pytorch_cos_sim function
        mock_cos_sim.return_value = torch.tensor([[0.8]])

        results = self.assembler.semantic_search(mock_context, 'test query', top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 3)  # source, similarity, chunk

    @patch('context_assembler.context_assembler.open')
    @patch('context_assembler.context_assembler.json.load')
    @patch('context_assembler.context_assembler.ContextAssembler.get_or_create_summary')
    def test_get_investment_overview(self, mock_get_or_create_summary, mock_json_load, mock_open):
        mock_json_load.return_value = {
            "id": "1",
            "name": "Test Investment",
            "folder_files": ["doc1.pdf"],
            "websites": ["https://example.com"]
        }
        
        mock_get_or_create_summary.side_effect = ['Summary of doc1', 'Summary of website']
        
        overview = self.assembler.get_investment_overview('1')

        self.assertIn('Test Investment', overview)
        self.assertIn('Summary of doc1', overview)
        self.assertIn('Summary of website', overview)

    @patch('context_assembler.context_assembler.ContextAssembler.semantic_search')
    def test_search_specific_document(self, mock_semantic_search):
        mock_context = {
            'documents': [{'file': 'doc1.pdf', 'chunks': ['This is a test chunk']}],
            'websites': [{'url': 'https://example.com', 'chunks': ['This is another test chunk']}]
        }

        mock_semantic_search.return_value = [('doc1.pdf', 0.8, 'This is a test chunk')]
        
        results = self.assembler.search_specific_document(mock_context, 'doc1.pdf', 'test query')

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 'doc1.pdf')

    def test_determine_sector(self):
        overview = "This is a real estate investment focused on residential properties."
        sector = self.assembler.determine_sector(overview)
        self.assertEqual(sector, "Residential Real Estate")

class TestSearchAllDocumentsTool(unittest.TestCase):
    def setUp(self):
        self.assembler = MagicMock()
        self.tool = SearchAllDocumentsTool(self.assembler)

    def test_run(self):
        self.assembler.assemble_context.return_value = {'test': 'context'}
        self.assembler.semantic_search.return_value = [('doc1.pdf', 0.8, 'Test chunk')]

        result = self.tool.run(investment_id='1', query='test query', top_k=1)

        self.assembler.assemble_context.assert_called_once_with('1', include_full_chunks=True)
        self.assembler.semantic_search.assert_called_once_with({'test': 'context'}, 'test query', 1)
        self.assertEqual(result, [('doc1.pdf', 0.8, 'Test chunk')])

class TestSearchSpecificDocumentTool(unittest.TestCase):
    def setUp(self):
        self.assembler = MagicMock()
        self.tool = SearchSpecificDocumentTool(self.assembler)

    def test_run(self):
        self.assembler.assemble_context.return_value = {'test': 'context'}
        self.assembler.search_specific_document.return_value = [('doc1.pdf', 0.8, 'Test chunk')]

        result = self.tool.run(investment_id='1', document_name='doc1.pdf', query='test query', top_k=1)

        self.assembler.assemble_context.assert_called_once_with('1')
        self.assembler.search_specific_document.assert_called_once_with({'test': 'context'}, 'doc1.pdf', 'test query', 1)
        self.assertEqual(result, [('doc1.pdf', 0.8, 'Test chunk')])

if __name__ == '__main__':
    unittest.main()