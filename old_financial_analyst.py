from crewai import Agent
from .base_agent import BaseAgent
import os
import json

class FinancialAnalyst(BaseAgent):
    def __init__(self, llm=None, knowledge_base_path="knowledge_bases/financial_analysis.txt"):
        super().__init__(
            name="Financial Analyst",
            role="Financial Analyst",
            goal="Analyze financial aspects of EB-5 investments. In particular, evaluate the financial viability and projections of the investment",
            backstory="Experienced financial analyst specializing in EB-5 investments with a strong background in financial modeling and ROI calculations.",
            allow_delegation=False,
            llm=llm
        )
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()

    def load_knowledge_base(self):
        with open(self.knowledge_base_path, 'r') as f:
            return f.read()

    def analyze(self, investment_context):
        self.set_context(investment_context)

        analysis_result = {}
        analysis = ""
        key_areas = [
            "capital structure",
            "financial projections",
            "valuation",
            "industry benchmarks",
            "historical performance",
            "economic factors"
        ]

        # 1. Start with knowledge base info
        analysis += f"**Financial Analysis based on Knowledge Base:**\n{self.knowledge_base}\n\n"

        # 2. Add relevant info from documents
        for area in key_areas:
            relevant_info = self.semantic_search(area)
            analysis_result[area] = self.analyze_area(area, relevant_info)
            analysis += f"**Relevant information on {area}:**\n"
            for _, chunk, source in relevant_info:
                analysis += f"- {chunk} (Source: {source})\n"
            analysis += "\n"

        # for area, result in analysis_result.items():
        #     if result.get('needs_deep_dive'):
        #         detailed_info = self.deep_dive(f"{area} in EB-5 investments")
        #         analysis_result[area]['deep_dive'] = self.analyze_deep_dive(area, detailed_info)

        blind_spots = self.identify_blind_spots(analysis_result)
        analysis_result['potential_blind_spots'] = blind_spots

        return analysis_result

    def analyze_area(self, area, relevant_info):
        summary = f"Analysis of {area} based on {len(relevant_info)} relevant pieces of information:\n"
        for _, chunk, source in relevant_info:
            summary += f"- {chunk[:100]}... (Source: {source})\n"
        
        return {
            'summary': summary,
            'needs_deep_dive': len(relevant_info) < 3
        }

    def analyze_deep_dive(self, area, detailed_info):
        return f"Detailed analysis of {area}:\n{detailed_info[:500]}..."

    def identify_blind_spots(self, analysis_result):
        blind_spots = []
        required_elements = [
            "cash flow projections",
            "risk factors",
            "exit strategy",
            "job creation estimates"
        ]
        for element in required_elements:
            if not any(element in result.get('summary', '') for result in analysis_result.values()):
                blind_spots.append(f"Missing information on {element}")
        return blind_spots

    def web_research(self, topic):
        search_results = self.web_search(f"EB-5 investment {topic}")
        relevant_info = [result for result in search_results if 'EB-5' in result['title']]
        return relevant_info