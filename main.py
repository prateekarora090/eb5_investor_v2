from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import logging
import json
from dotenv import load_dotenv
import argparse

# Import core utilities
from preprocessing.document_preprocessor import DocumentPreprocessor
from context_assembler.context_assembler import ContextAssembler, SearchAllDocumentsTool, SearchSpecificDocumentTool

# Import agents
from agents import Agents

# Import tasks
from tasks import create_financial_analyst_task, create_immigration_expert_task, create_risk_assessor_task, create_eb5_program_specialist_task 

# Import tools
from tools.pdf_reader import read_pdf
from tools.web_scraper import scrape_website
from tools.google_drive_reader import list_files_in_folder, read_file_from_drive

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
        return ChatGoogleGenerativeAI(model="gemini-pro")
    elif model_name == "gpt-3.5-turbo":
        return ChatOpenAI(model_name="gpt-3.5-turbo")
    # Add more model options as needed

def analyze_investments(investments, llm, report_name):
    # Processed input
    assembler = ContextAssembler('preprocessing/outputs/preprocessed_data')

    # Initialize tools (ones related to context assembler)
    search_all_docs_tool = SearchAllDocumentsTool(assembler)
    search_specific_doc_tool = SearchSpecificDocumentTool(assembler)
    
    # Create agents (4 specialist agents)
    agents = Agents(llm, search_all_docs_tool, search_specific_doc_tool)
    financial_analyst = agents.financial_analyst_agent()
    immigration_expert = agents.immigration_expert_agent()
    risk_assessor = agents.risk_assessor_agent()
    eb5_specialist = agents.eb5_program_specialist_agent()

    # Load personal information
    with open('secrets/eb5_personal_info.txt', 'r') as f:
        personal_info = f.read()

    # Create a report directory
    report_dir = os.path.join("outputs", report_name)
    os.makedirs(report_dir, exist_ok=True)

    results = []
    for investment in investments:
        # Grab an assembled context taking into acccount all of the provided input files
        context = assembler.assemble_context(investment['id'])
        investment_sector = assembler.determine_sector(assembler.get_investment_overview(investment['id']))
        investment_name = investment['name']

        # Create investment directory within the report directory
        investment_dir = os.path.join(report_dir, investment['name'])
        os.makedirs(investment_dir, exist_ok=True)

        # Check for existing analysis results
        analysis_results_file = os.path.join(investment_dir, "analysis_results.json")
        if os.path.exists(analysis_results_file):
            print(f"Loading existing analysis for {investment['name']} from {analysis_results_file}")
            with open(analysis_results_file, 'r') as f:
                results.append(json.load(f))
            continue  # Skip to the next investment

        # Create agent-specific tasks
        financial_analysis_task = create_financial_analyst_task(
            investment_name, investment_sector, financial_analyst, personal_info
        )
        immigration_expert_analysis_task = create_immigration_expert_task(
            investment_name, investment_sector, immigration_expert, personal_info
        )
        risk_assessment_analysis_task = create_risk_assessor_task(
            investment_name, investment_sector, risk_assessor, personal_info
        )
        eb5_program_compliance_analysis_task = create_eb5_program_specialist_task(
            investment_name, investment_sector, eb5_specialist, personal_info
        )

        # Crew
        crew = Crew(
            agents=[
                financial_analyst,
                immigration_expert,
                risk_assessor,
                eb5_specialist
                ],
            tasks=[
                financial_analysis_task,
                immigration_expert_analysis_task,
                risk_assessment_analysis_task,
                eb5_program_compliance_analysis_task
                ],
            verbose=2
        )

        # Run the crew
        result = crew.kickoff()

        # Save the analysis results
        print(f"Saving analysis for {investment['name']} to {analysis_results_file}")
        with open(analysis_results_file, 'w') as f:
            json.dump(result, f, indent=4)

        results.append(result)
    print("Completed!")

def main():
    parser = argparse.ArgumentParser(description="EB-5 Investment Analysis")
    parser.add_argument("action", choices=["preprocess", "analyze"], help="Action to perform")
    parser.add_argument("--report_name", help="Name of the report (used for output directory)", default="eb5_analysis")
    args = parser.parse_args()

    if args.action == "preprocess":
        print("Starting preprocessing. Check 'preprocessing.log' for progress.")
        preprocessor = DocumentPreprocessor()
        log_file = os.path.join('preprocessing', 'outputs', 'preprocessing.log')
        preprocessor.preprocess_investments('inputs/options.json')
        
    elif args.action == "analyze":
        print("~~ Starting analysis phase ~~~")
        # Get llm
        llm = get_llm("gpt-3.5-turbo")  # Use this for testing
        # llm = get_llm("gemini-pro")  # Use this for final runs
        
        # Read investment options from JSON file
        with open('inputs/options.json', 'r') as f:
            all_investments = json.load(f)
        
        # Only consider the first option for initial testing
        investments_to_analyze = all_investments[:1]
        print(f"Analyzing {len(investments_to_analyze)} investment options")

        # Wrap main functionality in try-except
        try:
            result = analyze_investments(investments_to_analyze, llm, args.report_name)
            print(result)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"An error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()

