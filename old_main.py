from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
import os
import logging
import json
from dotenv import load_dotenv
import argparse

# Import core utilities
from preprocessing.document_preprocessor import DocumentPreprocessor
from context_assembler.context_assembler import ContextAssembler

# Import agents
from agents.base_agent import BaseAgent
from agents.document_processor import DocumentProcessor
from agents.financial_analyst import FinancialAnalyst
from agents.immigration_law_expert import ImmigrationLawExpert
from agents.risk_assessor import RiskAssessor
from agents.market_research_specialist import MarketResearchSpecialist
from agents.eb5_program_specialist import EB5ProgramSpecialist
from agents.investment_comparator import InvestmentComparator

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

def get_llm(model_name="gemini-pro"):
    if model_name == "gemini-pro":
        return ChatGoogleGenerativeAI(model="gemini-pro")
    elif model_name == "gpt-3.5-turbo":
        return OpenAI(model_name="gpt-3.5-turbo")
    # Add more model options as needed

def analyze_investments(investments, llm):
    # Initialize agents
    # document_processor = DocumentProcessor(llm=llm)
    # market_researcher = MarketResearchSpecialist(llm=llm)

    assembler = ContextAssembler('preprocessing/outputs/preprocessed_data')
    agents = {
        'financial_analyst': FinancialAnalyst(llm=llm, knowledge_base_path="knowledge_bases/financial_analysis.txt"),
        'immigration_expert': ImmigrationLawExpert(llm=llm, knowledge_base_path="knowledge_bases/immigration_law.txt"),
        'risk_assessor': RiskAssessor(llm=llm, knowledge_base_path="knowledge_bases/risk_assessment.txt"),
        'eb5_specialist': EB5ProgramSpecialist(llm=llm, knowledge_base_path="knowledge_bases/eb5_program.txt")
    }

    results = []
    for investment in investments:
        # Grab an assembled context taking into acccount all of the provided input files
        context = assembler.assemble_context(investment['id'])
        
        analysis_task = Task(
                description=f"Analyze financial viability and projects of investment {investment['id']}",
                agent=agents['financial_analyst'],
                context=context
            )

        # TODO: Uncommment this once we have other analysts
        # tasks = [
        #     Task(
        #         description=f"Analyze financial viability and projects of investment {investment['id']}",
        #         agent=agents['financial_analyst'],
        #         context=context
        #     ),
        #     Task(
        #         description=f"""Evaluate compliance with immigration laws and EB-5 program requirements for {investment['id']} with a special focus on
        #         with a special focus on law compliance, including investor eligibility, source of funds issues, and general USCIS policies.""",
        #         agent=agents['immigration_expert'],
        #         context=context
        #     ),
        #     Task(
        #         description=f"Identify and assess potential risks for investment {investment['id']}",
        #         agent=agents['risk_assessor'],
        #         context=context
        #     ),
        #     Task(
        #         description=f"""Evaluate EB-5 program compliance for investment {investment['id']},  with a special focus on
        #         program requirements, project structuring and regional center compliance.""" 
        #         agent=agents['eb5_specialist'],
        #         context=context
        #     )
        # ]

        # TODO: Uncommment this once we have other analysts
        # crew = Crew(
        #     agents=list(agents.values()),
        #     tasks=tasks,
        #     verbose=2
        # )
        # result = crew.kickoff()

        ## Commented stuff above, instead have:
        result = analysis_task.execute()
        
        results.append(result)

    # TODO: Uncommment this once we have a comparison analyst!
    # Add final comparison task
    # comparison_task = Task(
    #     description="Compare and rank all analyzed investments",
    #     agent=InvestmentComparator(llm=llm),
    #     context=results
    # )
    # final_result = comparison_task.execute()
    # return final_result
    print("Completed!")

def main():
    parser = argparse.ArgumentParser(description="EB-5 Investment Analysis")
    parser.add_argument("action", choices=["preprocess", "analyze"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "preprocess":
        print("Starting preprocessing. Check 'preprocessing.log' for progress.")
        preprocessor = DocumentPreprocessor()
        log_file = os.path.join('preprocessing', 'outputs', 'preprocessing.log')
        preprocessor.preprocess_investments('inputs/options.json')
        
    elif args.action == "analyze":
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
            result = analyze_investments(investments_to_analyze, llm)
            print(result)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"An error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()

