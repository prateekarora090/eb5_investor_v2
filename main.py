from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import logging
from ollama_wrapper import OllamaWrapper
import json
from dotenv import load_dotenv
import argparse

# Import config values
import config

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

# TODO: Ensure config's values are honored

def get_llm(model_enum="local-llama"):
    if model_enum == "gemini-pro":
        return ChatGoogleGenerativeAI(model_name="gemini-pro")
    elif model_enum == "gpt-3.5-turbo":
        return ChatOpenAI(model_name="gpt-3.5-turbo")
    elif model_enum == "local--llama":
        return OllamaWrapper(model_name="llama3:8b-instruct-q8_0")
    # Add more model options as needed
    else:
        raise ValueError(f"Invalid model name: {model_enum}")

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
    with open('secrets/eb5_personal_info.txt', encoding='utf-8') as f:
        personal_info = f.read()

    # Create a report directory
    report_dir = os.path.join("outputs", report_name)
    os.makedirs(report_dir, exist_ok=True)

    results = []
    for investment in investments:
        # Grab an assembled context taking into acccount all of the provided input files
        context = assembler.assemble_context(investment['id'])
        investment_name = investment['name']
        investment_id = investment['id']
        investment_overview = assembler.get_investment_overview(investment['id'])

        # Create investment directory within the report directory
        investment_dir = os.path.join(report_dir, investment['name'])
        os.makedirs(investment_dir, exist_ok=True)

        # Check for existing analysis results
        analysis_results_file = os.path.join(investment_dir, "analysis_results.json")
        log_file_path = os.path.join(investment_dir, "log.txt")
        if os.path.exists(analysis_results_file):
            print(f"Loading existing analysis for {investment['name']} from {analysis_results_file}")
            with open(analysis_results_file, 'r') as f:
                results.append(json.load(f))
            continue  # Skip to the next investment

        # Task specific output files
        financial_analysis_output_file = os.path.join(investment_dir, "financial_analysis_results.json")
        immigration_expert_output_file = os.path.join(investment_dir, "immigration_expert_results.json")
        risk_assessment_output_file = os.path.join(investment_dir, "risk_assessment_results.json")
        eb5_program_compliance_output_file = os.path.join(investment_dir, "eb5_program_compliance_results.json")

        # Create agent-specific tasks
        financial_analysis_task = create_financial_analyst_task(
            investment_id, investment_name, investment_overview, financial_analyst, 
            personal_info, financial_analysis_output_file
        )
        immigration_expert_analysis_task = create_immigration_expert_task(
            investment_id, investment_name, investment_overview, immigration_expert,
            personal_info, immigration_expert_output_file
        )
        risk_assessment_analysis_task = create_risk_assessor_task(
            investment_id, investment_name, investment_overview, risk_assessor,
            personal_info, risk_assessment_output_file
        )
        eb5_program_compliance_analysis_task = create_eb5_program_specialist_task(
            investment_id, investment_name, investment_overview, eb5_specialist,
            personal_info, eb5_program_compliance_output_file
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
            verbose=True,
            output_log_file=log_file_path, # output to log file
            full_output=True, # didn't work for me, but each task has output_file too.
            process=Process.sequential,
            memory=True,
        )

        # Run the crew
        result = crew.kickoff()

        # Save the analysis results
        print(f"Saving analysis for {investment['name']} to {analysis_results_file}")
        with open(analysis_results_file, 'w') as f:
            with open(analysis_results_file, 'w') as f:
                result_dict = {
                    "raw": result.get('raw', ''),
                    "json_dict": result.get('json_dict', {}),
                    "tasks_output": [
                        {
                            "task_id": task.get('task_id', ''),
                            "output": task.get('output', ''),
                            "agent_name": task.get('agent_name', '')
                        } for task in result.get('tasks_output', [])
                    ],
                    "token_usage": result.get('token_usage', {})
                }
                json.dump(result_dict, f, indent=4)
                print(f"{investment['name']} analysis completed! \n")
                print(f"Results: {result_dict}")

        results.append(result)
    print("Completed!")

def main():
    parser = argparse.ArgumentParser(description="EB-5 Investment Analysis")
    parser.add_argument("action", choices=["preprocess", "testing", "abstract", "analyze"], help="Action to perform")
    parser.add_argument("--report_name", help="Name of the report (used for output directory)", default="eb5_analysis")
    args = parser.parse_args()

    # 1st preprocess (extract) phase for the inputted documents
    # This phase extracts text and visual content from the inputted documents and stores it in a json file
    # This json file is used as input in the next phase
    if args.action == "preprocess":
        print("Starting preprocessing. Check 'preprocessing.log' for progress.")
        preprocessor = DocumentPreprocessor()
        log_file = os.path.join('preprocessing', 'outputs', 'preprocessing.log')
        preprocessor.preprocess_investments('inputs/options.json')

    # 2nd preprocess (abstract) phase for the inputted documents
    # This phrase reads the json file generated by the previous phase to generate summaries.
    # TODO: Define this phase better, after modularizing the code, also add to README.
    elif args.action == "abstract":
        print("~~ Starting analysis phase ~~~")
        assembler = ContextAssembler('preprocessing/outputs/preprocessed_data')
        investment_id = '1'
        result = ""

        for i in range(1, 12):
            investment_id = str(i)
            print (f"~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n") 

            print (f"~ Processing investment {investment_id} \n")
            investment_overview = assembler.get_investment_overview(investment_id)
            print(f"Investment Overview: {investment_overview}")
           
            print (f"~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n") 

        print(f" Finished 2nd preprocessing stage \n")
        print(f"~ ~ ~~~~ ~~~ ~~~~ ~~~ ~~~~ ~~~ ~~~~ ~~~ ~~~~ ~~~ ~~~~ ~~~ \n")

    elif args.action == "analyze":
        print("~~ Starting analysis phase ~~~")
        # Get llm
        llm = get_llm("local--llama")  # Use this for testing
        # llm = get_llm("gpt-3.5-turbo")  # Use this for testing
        # llm = get_llm("gemini-pro")  # Use this for final runs
        
        # Read investment options from JSON file
        with open('inputs/options.json', 'r') as f:
            all_investments = json.load(f)
        
        # Only consider the first option for initial testing
        investments_to_analyze = all_investments
        print(f"Analyzing {len(investments_to_analyze)} investment options")

        # Wrap main functionality in try-except
        try:
            result = analyze_investments(investments_to_analyze, llm, args.report_name)
            print(result)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"An error occurred. Please check the log file for details.")
    
    elif args.action == "testing":
        print("~~~~ Testing analysis workflows (i.e. preprocessed output makes tools work!)")
        # Read preprocessed output, initiate assembler
        assembler = ContextAssembler('preprocessing/outputs/preprocessed_data')
        investment_id = '1'
        result = ""

        # # Test #1: Generate investment context
        investment_context = assembler.assemble_context(investment_id)
        investment_context_with_chunks = assembler.assemble_context(investment_id, include_full_chunks=True)
        
        result += f"-------------------------------------- \n"
        result += f"Test #1: Generating investment context \n"
        result += f"-------------------------------------- \n"
        result += f"\n"
        result += f"Generated investment context = {investment_context} \n"
        result += f"\n"
        result += f"-------------------------------- \n"
        
        # # Test #2: Search all docs
        all_docs_query = "financial projections ROI risks"

        result += f"-------------------------------------- \n"
        result += f"Test #2: Searching all docs (query={all_docs_query}) \n"
        result += f"-------------------------------------- \n"
        search_all_docs_tool = SearchAllDocumentsTool(assembler)
        all_doc_result = assembler.semantic_search(investment_context_with_chunks, all_docs_query, 2) # mimics SearchAllDocumentsTool._run()
        formatted_all_doc_result = json.dumps(all_doc_result, indent=4)

        result += f"Result = {formatted_all_doc_result} \n"
        result += f"\n"
        result += f"-------------------------------- \n"

        # # Test #3: Search specific doc tool
        search_specific_doc_tool = SearchSpecificDocumentTool(assembler)
        specific_query = "investor eligibility"
        specific_doc = "Subscription Booklet"

        result += f"-------------------------------------- \n"
        result += f"Test #3: Searching specific doc (query={specific_query} & doc={specific_doc}) \n"

        result += f"-------------------------------------- \n"
        
        specific_doc_results = assembler.search_specific_document(
            investment_context_with_chunks, specific_doc, specific_query, 5) # mimics SearchSpecificDocumentTool._run()
        formatted_specific_doc_results = json.dumps(specific_doc_results, indent=4)

        result += f"Result = {formatted_specific_doc_results} \n"
        result += f"\n"
        result += f"-------------------------------- \n"
        
        # # Test #4: Load personal information
        result += f"-------------------------------------- \n"
        result += f"Test #4: Load personal information \n"
        result += f"-------------------------------------- \n"
        with open('secrets/eb5_personal_info.txt', encoding='utf-8') as f:
            personal_info = f.read()
        
        result += f"Result = {personal_info} \n"
        result += f"\n"
        result += f"-------------------------------- \n"

        # Test #5: Test investment overview
        investment_overview = assembler.get_investment_overview(investment_id)
        result += f"-------------------------------------- \n"
        result += f"Test #5: Determine investment overview \n"
        result += f"-------------------------------------- \n"
        result += f"Result = {investment_overview} \n"
        result += f"\n"
        result += f"-------------------------------- \n"

        # Test #6: Evaluate investment sector
        investment_sector = assembler.determine_sector(investment_overview)
        result += f"-------------------------------------- \n"
        result += f"Test #6: Evaluate investment sector \n"
        result += f"-------------------------------------- \n"
        result += f"Result = {investment_sector} \n"
        result += f"\n"
        result += f"-------------------------------- \n"

        # Print all results!!!
        print(result)

if __name__ == "__main__":
    main()
