from crewai import Agent
from tools.web_search_tool import WebSearchTool
from tools.web_scraper_tool import WebScraperTool 
from crewai_tools import TextSearchTool 

#TODO: Implement a tool to give agents access to "historical data" -- this would be shared knowledge they all should have!!
## ^^ This can rely on certain websites they can visit or scrape, it can also be indexed / we can create embedding vectors for it as well.

class Agents:

    def __init__(self, llm, search_all_documents_tool, search_specific_document_tool):
        self.llm = llm
        self.web_search_tool = WebSearchTool()
        self.web_scraper_tool = WebScraperTool()
        self.search_all_documents_tool = search_all_documents_tool
        self.search_specific_document_tool = search_specific_document_tool
        self.knowledge_base_dir = knowledge_base_dir

    def financial_analyst_agent(self):
        knowledge_base = self._load_knowledge_base("financial_analysis.txt")
        knowledge_search_tool = TextSearchTool(knowledge_base)
        
        return Agent(
            name="Financial Analyst",
            role="Financial Analyst",
            goal="""Analyze financial aspects of EB-5 investments. In particular, evaluate the
             financial viability and projections of the investment""",
            backstory="""Experienced financial analyst specializing in EB-5 investments with a strong background in
             financial modeling and ROI calculations. **Particularly adept at analyzing capital structures, 
             financial projections, and identifying potential financial risks.**""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=[
                self.web_search_tool, 
                self.web_scraper_tool,
                self.search_all_documents_tool, 
                self.search_specific_document_tool,
                knowledge_search_tool]
            # TODO: memory?
        )
    
    def eb5_program_specialist_agent(self):
        knowledge_base_path = os.path.join(self.knowledge_base_dir, "eb5_program.txt")
        knowledge_search_tool = TextSearchTool(knowledge_base_path)

        return Agent(
            name="EB-5 Program Specialist",
            role="EB-5 Program Specialist",
            goal="Provide deep insights into EB-5 program requirements and nuances",
            backstory="""Expert in EB-5 program regulations with years of experience in successful EB-5 project
             implementations. **Possesses deep knowledge of TEA designation criteria, job creation methodologies, and regional center compliance.**""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=[
                self.web_search_tool,
                self.web_scraper_tool,
                self.search_all_documents_tool,
                self.search_specific_document_tool,
                knowledge_search_tool 
            ]
            # TODO: memory?
        )
    
    def immigration_expert_agent(self):
        knowledge_base_path = os.path.join(self.knowledge_base_dir, "immigration_law.txt")
        knowledge_search_tool = TextSearchTool(knowledge_base_path)

        return Agent(
            name="Immigration Law Expert",
            role="Immigration Law Expert",
            goal="""Evaluate the investment's compliance with US immigration laws, focusing
             on investor eligibility and source of funds.""",
            backstory=backstory="""Experienced risk management professional, specializing in analyzing and mitigating risks in
             EB-5 investment projects. **Proven ability to identify financial, legal, market, and developer-related risks
              in EB-5 investments, drawing upon historical data and industry trends.**""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=[
                self.web_search_tool,
                self.web_scraper_tool,
                self.search_all_documents_tool,
                self.search_specific_document_tool,
                knowledge_search_tool 
            ]
            # TODO: memory?
        )
    
    def risk_assessor_agent(self):
        knowledge_base_path = os.path.join(self.knowledge_base_dir, "risk_assessment.txt")
        knowledge_search_tool = TextSearchTool(knowledge_base_path)

        return Agent(
            name="Risk Assessor",
            role="Risk Assessor",
            goal="Identify and assess potential risks associated with the EB-5 investment.",
            backstory="""Experienced risk management professional, specializing in analyzing and mitigating risks in
             EB-5 investment projects.""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=[
                self.web_search_tool,
                self.web_scraper_tool,
                self.search_all_documents_tool,
                self.search_specific_document_tool,
                knowledge_search_tool 
            ]
            # TODO: memory?
        )

    def _load_knowledge_base(self, file_name):
        """Loads the knowledge base from the specified file."""
        with open(os.path.join(self.knowledge_base_dir, file_name), 'r') as f:
            return f.read()