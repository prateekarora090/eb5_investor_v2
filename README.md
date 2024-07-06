# EB-5 Investment Analysis with CrewAI

This project uses the CrewAI framework to analyze EB-5 investment opportunities from multiple perspectives. It leverages a team of specialized AI agents to evaluate the financial, legal, risk, and EB-5 program compliance aspects of each investment.

## Project Overview

The system consists of the following key components:

**1. Preprocessing:**
   - `preprocessing/document_preprocessor.py`:  Processes raw investment documents (PDFs and websites) and generates embeddings for efficient semantic search. 
   - Outputs are stored in the `preprocessing/outputs/preprocessed_data` directory.

**2. Context Assembly:**
   - `context_assembler/context_assembler.py`: Assembles the context for each investment from the preprocessed data. 
   - Provides tools for:
     -  `SearchAllDocumentsTool`:  Searches across all documents within an investment context.
     -  `SearchSpecificDocumentTool`:  Searches within a specific document.
     -  `get_investment_overview`: Generates a summary of the investment and determines the investment sector.

**3. Agents:**
   - `agents.py`:  Defines the specialized AI agents:
     -  `FinancialAnalyst`: Evaluates the financial viability of investments.
     -  `ImmigrationExpert`:  Assesses compliance with immigration laws and EB-5 program requirements.
     -  `RiskAssessor`:  Identifies and analyzes potential risks. 
     -  `EB5ProgramSpecialist`: Evaluates the investment's alignment with the EB-5 program's goals and requirements.

**4. Tasks:**
   - `tasks.py`: Creates Task objects for each agent, providing detailed descriptions, expected outputs, and access to tools.

**5. Tools:**
   - `tools/`: Contains various tools used by the agents, including:
      -  `KnowledgeSearchTool`: Searches the agents' knowledge bases.
      -  `WebSearchTool`:  Searches the web for information using a search API.
      -  `WebScraperTool`: Scrapes content from websites. 

**6. Main Analysis Script:**
   - `main.py`:  Orchestrates the analysis process:
      -  Loads investment data.
      -  Initializes the `ContextAssembler`, tools, and agents.
      -  Creates tasks for each agent and investment.
      -  Runs the analyses using the CrewAI framework.
      -  Persists analysis results to JSON files.

## Usage

**1. Preprocess Investment Data:**

```bash
python main.py preprocess
```

**2. Analyze Investments:**

```bash
python main.py analyze --report_name <report_name>
```

Replace `<report_name>` with a descriptive name for your analysis run (e.g., "first_run", "2023-11-analysis").

**Output:**

Analysis results for each investment are saved in JSON files within the `outputs/<report_name>` directory.

## Fast-Follows (Planned Enhancements)

- **Historical EB-5 Stats Tool:** Create a tool that provides access to a structured database of historical EB-5 case data. This will provide shared knowledge to all agents about past trends, approvals, denials, and common issues.
- **Code Composition:** Improve code organization by moving the main analysis logic from `main.py` to a dedicated `analysis/` directory, similar to the `preprocessing/` structure.  

## Future Development

- **Investment Ranking and Comparison:**  Implement logic to rank investments based on the agents' analyses and provide an overall comparison.
- **Reporting and Visualization:** Develop a system to generate user-friendly reports and visualizations of the analysis results.
- **Knowledge Base Expansion and Refinement:** Continuously expand and refine the agents' knowledge bases with more detailed information and insights. 

## Contributing

Contributions and suggestions for improvement are welcome! Please feel free to open issues or pull requests on the project's GitHub repository. 
