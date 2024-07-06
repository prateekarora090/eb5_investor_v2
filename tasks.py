from crewai import Task

def create_financial_analyst_task(investment_name, investment_sector, agent, context, personal_info):
    """Creates a Task object for the Financial Analyst."""
    return Task(
        description=f"""
        You are a skilled financial analyst specializing in EB-5 investments. 
        Your task is to thoroughly analyze the financial aspects of the following EB-5 investment:

        **Investment Name:** {investment_name}
        **Investment Sector:** {investment_sector}
        **Personal Information:** {personal_info}

        You have access to the following tools:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - Knowledge Search: Allows you to search a knowledge base for information about financial analysis and EB-5 investments. 
        - Web Search: Allows you to search for relevant information on the web.
        - Web Scraper: Allows you to scrape content from a website.

        **Instructions:**
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" and "Search Specific Document" tools.** Pay close attention to financial statements, projections, and any details related to the capital structure, ROI, or potential risks. 
        2. Use "Knowledge Search" to refresh your knowledge about common financial risks, important metrics, and EB-5 specific financial considerations.
        3. Use Web Search to find information about the developers or sponsors and understand their track record.
        4. Use Web Scraper to get more context about websites returned by Web Search.
        5. Analyze the investment's financial viability, projections, and potential risks. 
        6. Identify any potential blind spots or missing information that requires further investigation.

        **Provide a comprehensive financial analysis of the investment, including your findings, insights, and recommendations.** 
        """,
        expected_output="""Any valid output must also:
            1) Must focus on all of the following key areas: 
                - i) capital structure, 
                - ii) financial projections,
                - iii) valuation
                - iv) industry benchmarks
                - v) historical performance
                - vi) economic factors
            2) Must watch out for blind spots like:
                - i) cash flow projections
                - ii) risk factors
                - iii) exit strategy
                - iv) job creation estimates
            """,
        agent=agent,
        context=context
    )

def create_immigration_expert_task(investment_name, investment_sector, agent, context, personal_info):
    """Creates a Task object for the Immigration Law Expert."""
    return Task(
        description=f"""
        You are an experienced immigration lawyer specializing in the EB-5 program.
        Your task is to evaluate the following EB-5 investment for compliance with immigration laws and program requirements:

        **Investment Name:** {investment_name}
        **Investment Sector:** {investment_sector}
        **Personal Information:** {personal_info}

        You have access to:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - Knowledge Search: Search a knowledge base for immigration laws, EB-5 regulations, and USCIS policies.
        - Web Search: Research recent legal updates, precedent decisions, and USCIS announcements.

        **Instructions:**
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" and "Search Specific Document" tools.** Pay close attention to details related to investor eligibility, job creation methodologies, TEA designation, source of funds, and regional center involvement (if applicable).
        2. Use "Knowledge Search" to review key legal requirements for EB-5 investors and projects.
        3. Scrutinize the investment documents for potential compliance issues:
           - Investor eligibility: Source of funds, background checks, etc.
           - Job creation: Methodology, documentation, TEA designation compliance.
           - Project structure: Alignment with EB-5 program rules, regional center involvement (if applicable).
        4. Utilize "Web Search" to research recent USCIS policy changes, precedent decisions, or any relevant legal updates.

        **Deliver a comprehensive legal analysis of the investment, highlighting any potential compliance concerns, risks, or required clarifications.**
        """,
        expected_output="""Any valid output must:
        1) Evaluate compliance in the following key areas:
            - Investor eligibility, including source of funds documentation
            - Job creation methodology and adherence to EB-5 requirements
            - Project structure and its compliance with program rules
            - Regional Center compliance (if applicable)
        2) Identify potential red flags or areas of concern that require further investigation.
        3) Provide clear recommendations or next steps for addressing any compliance issues.
        """,
        agent=agent,
        context=context
    )

def create_risk_assessor_task(investment_name, investment_sector, agent, context, personal_info):
    """Creates a Task object for the Risk Assessor."""
    return Task(
        description=f"""
        You are a risk management expert specializing in EB-5 investments.
        Analyze the following investment for potential risks and red flags:

        **Investment Name:** {investment_name}
        **Investment Sector:** {investment_sector}
        **Personal Information:** {personal_info}

        Tools at your disposal:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - Knowledge Search: Access a risk assessment knowledge base for EB-5 investments.
        - Semantic Search: Search investment documents for risk-related information.
        - Search Specific Document: Focus your search on a specific document.
        - Web Search: Research industry trends, market risks, and developer/sponsor reputation.

        **Instructions:**
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" and "Search Specific Document" tools.** Look for any red flags related to the financial projections, capital structure, market conditions, developer experience, legal compliance, or exit strategy.
        2. Begin by using "Knowledge Search" to review common EB-5 investment risks.
        3. Thoroughly analyze the investment documents for potential red flags:
           - Financial: Unrealistic projections, weak capital structure, insufficient collateral, etc.
           - Legal: Compliance issues, incomplete documentation, unclear escrow arrangements.
           - Market: Sector volatility, competition, economic downturn risks.
           - Developer/Sponsor:  Lack of experience, poor track record, financial instability.
        4. Use "Web Search" to gather additional information about the sector, market trends, and the developer/sponsor's reputation.

        **Deliver a comprehensive risk assessment, highlighting key areas of concern, potential mitigation strategies, and an overall risk rating for the investment.**
        """,
        expected_output="""The risk assessment must include:
        1) A clear identification of all major risks, categorized by type (e.g., financial, legal, market, developer/sponsor).
        2) An evaluation of the severity of each risk, considering likelihood and potential impact.
        3)  Proposed mitigation strategies for the most significant risks.
        4) An overall risk rating for the investment (e.g., low, medium, high).
        """,
        agent=agent,
        context=context
    )

def create_eb5_program_specialist_task(investment_name, investment_sector, agent, context, personal_info):
    """Creates a Task object for the EB-5 Program Specialist."""
    return Task(
        description=f"""
        You are an expert in the EB-5 program, well-versed in its nuances and requirements.
        Your task is to evaluate the following investment's alignment with the program:

        **Investment Name:** {investment_name}
        **Investment Sector:** {investment_sector}
        **Personal Information:** {personal_info}

        You have access to the following tools:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - Knowledge Search: Explore a knowledge base containing detailed EB-5 program information.
        - Web Search:  Research recent EB-5 program updates, policy changes, and successful project examples.

        **Instructions:**
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" and "Search Specific Document" tools.**  Focus on the project structure, TEA designation, job creation methodology, and any regional center involvement.
        2. Begin by using "Knowledge Search" to refresh your understanding of EB-5 program requirements, TEA designations, job creation methodologies, and regional center compliance (if applicable).
        3. Analyze the investment documents:
           - Project structure: How does it align with the EB-5 program's goals?
           - TEA designation: Is it justified and properly documented?
           - Job creation: Is the methodology sound and supported by reliable projections?
           - Regional center (if applicable):  Is it reputable and compliant?
        4. Use "Web Search" to find comparable projects and understand their success stories or challenges.

        **Provide an in-depth evaluation of how well this investment aligns with the EB-5 program, highlighting any strengths, weaknesses, or areas requiring further scrutiny.**
        """,
        expected_output="""The evaluation should address:
        1) Alignment of the project structure with the EB-5 program's goals.
        2) Justification and documentation of the TEA designation.
        3) Soundness and reliability of the job creation methodology.
        4)  Reputability and compliance of the regional center (if applicable).
        5)  Overall strengths, weaknesses, and areas requiring further attention.
        """,
        agent=agent,
        context=context
    )