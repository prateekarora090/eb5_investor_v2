from crewai import Task

def create_financial_analyst_task(investment_id, investment_name, investment_overview, agent, personal_info, output_file):
    """Creates a Task object for the Financial Analyst."""
    return Task(
        description=f"""
        # Goal
        You are a skilled financial analyst specializing in EB-5 investments. 
        You have been hired by two investors to thoroughly analyze the financial aspects of an EB-5 opportunity.
            - **Investment ID:** {investment_id}
            - **Investment Name:** {investment_name}

        # Introductory Information
        ## Client Information
        Here's some personal background about the two investors that hired you: {personal_info}
        Feel free to ask any other questions that may assist your assessment.

        ## Investment Overview
        Here's an investment overview and a corpus of docs provided by the fund manager. Pay close attention
        to this broad overview because you'll also be able to search these docs for further information.

        Here's the investment overview: {investment_overview}

        # Research Tools
        You have access to the following tools:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - **Knowledge Search**: Allows you to search a knowledge base for information about financial analysis and EB-5 investments. 
        - **Web Search**: Allows you to search for relevant information on the web.
        - **Web Scraper**: Allows you to scrape content from a website.

        # Instructions
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" 
            and "Search Specific Document" tools.** Pay close attention to financial statements, projections,
            and any details related to the capital structure, ROI, or potential risks. 
        2. Use "Knowledge Search" to refresh your knowledge about common financial risks, important metrics
            and EB-5 specific financial considerations.
        3. Use Web Search to find information about the developers or sponsors and understand their track record.
        4. Use Web Scraper to get more context about websites returned by Web Search.
        5. Analyze the investment's financial viability, projections, and potential risks. 
        6. Identify any potential blind spots or missing information that requires further investigation.

        # Core Deliverable

        ## Stylistic Suggestion / General Tips
        - Please ensure your analysis is detailed enough to summarize ALL important insights from your research.
        - Please ensure that you **use appropriate formatting** to present your findings.
        - Note that investors do NOT care about rate of return, but rather the return OF their capital

        ## Hard Requirements
        - Provide a **comprehensive financial analysis of the investment, including your findings, insights, and recommendations.** 
            - Start with a table summarizing key metrics that should include:
                - 1) Place in Capital Stack
                - 2) Principal Repayment Timeline (including any extensions)
                - 3) % EB-5 Capital of Total Capital
                - Optionally: Category (Residential or Commercial or Hybrid), Location (tax rate, business-friendliness, etc.)
        - Ensure that you **cite your sources** for all important information backing insights, e.g. for valuations.
        - Provide hard nunbers wherever possible. Non-negotiable for capital projections, valuation, etc.
        - Specific exit strategy: 
            - Should be based on hard numbers. 
            - Capital projections like revenue MUST take into account capital financed by EB-5 and capital structure.
            - Evaluate feasibility and risk specific to the investment. Specify those details.
            - Should also consider demand by evaluating cost of living, business friendliness and tax climate based on the location.
        - Follow-ups:
            - Add any follow-up's to consider -- questions for investment team, another expert to consult, etc.
            - Do NOT recommend action items that can be researched. Please spend the time to thoroughly research with available tools.
            - Strongly prefer follow-up's mention specific blind spots and specific missing information 
                - e.g. don't say "unclear exit strategy", say "exit strategy evaluation requires clarity on capital structure"
        - For risks: Do NOT mention generic risks without corroborating with information specific to the investment with cited sources.
        - Conclusion: Assign a financial robustness score for EB5 loan repayment: "low", "medium" or "high" score.

            
        """,
        expected_output="""Any valid output must:
            1) Must focus on all of the following key areas: 
                - i) capital structure, 
                - ii) financial projections (including DCF, cash flow, ROI, etc.),
                - iii) valuation (including relevant industry benchmarks, historical performance & economic factors)
                - iv) financial risks (property type, liquidity,  geographic factors like tax rate, business-friendliness, etc.), 
                - v) exit strategy analysis (investment horizon, capital structure, etc.)
            2) Must watch out for any blind spots:
                - Any specific missing information for above analysis (cash flow, cap structure, etc.)
            3) Summary / Conclusion
                - Stats must be clearly stated, ideally in a table.
                - Conclusion with financial robustness score
            """,
        agent=agent,
        output_file=output_file
    )

def create_immigration_expert_task(investment_id, investment_name, investment_overview, agent, personal_info, output_file):
    """Creates a Task object for the Immigration Law Expert."""
    return Task(
        description=f"""
        # Goal
        You are an experienced immigration lawyer specializing in the EB-5 program.
        You have been hired by two investors to thoroughly evaluate an EB-5 investment
        for compliance with immigration laws and program requirements.
            - **Investment ID:** {investment_id}
            - **Investment Name:** {investment_name}

        # Introductory Information
        ## Client Information
        Here's some personal background about the two investors that hired you: {personal_info}
        Feel free to ask any other questions that may assist your assessment.

        ## Investment Overview
        Here's an investment overview and a corpus of docs provided by the fund manager. Pay close attention
        to this broad overview because you'll also be able to search these docs for further information.

        Here's the investment overview: {investment_overview}

        # Research Tools
        You have access to the following tools:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - **Knowledge Search**: Search a knowledge base for immigration laws, EB-5 regulations, and USCIS policies.
        - **Web Search**: Research recent legal updates, precedent decisions, and USCIS announcements.
        - **Web Scraper**: Allows you to scrape content from a website.
        
        # Instructions
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" and "Search Specific Document" tools.** Pay close attention to details related to investor eligibility, job creation methodologies, TEA designation, source of funds, and regional center involvement (if applicable).
        2. Use "Knowledge Search" to review key legal requirements for EB-5 investors and projects.
        3. Scrutinize the investment documents for potential compliance issues:
           - TEA designation compliance
           - Job creation: Methodology, timing, documentation, financial viability. Please cite sources.
           - Project structure: Alignment with EB-5 program rules, regional center involvement (if applicable).
           - Other compliance risks including reputational risks
        4. Utilize "Web Search" to research recent USCIS policy changes, precedent decisions, or any relevant legal updates.
        5. Use "Web Scraper" to get more context about websites returned by Web Search.

        # Deliverable
        - Deliver a **comprehensive legal analysis of the investment, highlighting any potential compliance concerns, risks, or required clarifications.**
        - Please ensure that you **cite your sources** for all important information that you use.
        - Please ensure your analysis is detailed enough to summarize ALL important insights from your research.
        - Please ensure that you **use appropriate formatting** to present your findings.
        - Importantly, please add any follow-up's to consider -- questions for investment team, another expert to consult, etc.
            - Do NOT recommend action items that can be researched. Please spend the time to thoroughly research with available tools.
            - Strongly prefer follow-up's mention specific blind spots and specific missing information to keep them actionable. 
        """,
        expected_output="""Any valid output must:
        1) Evaluate compliance in the following key areas:
            - Job creation state, methodology and EB-5 compliance.
                - Have the jobs already been created? Do they seem viable?
                - Please provide hard numbers. Must cite sources!
            - TEA designation compliance
            - Project structure and its compliance with program rules
            - Regional Center compliance (if applicable)
        2) Identify potential red flags or areas of concern that require further investigation.
        3) Provide clear recommendations or next steps for addressing any compliance issues.
        """,
        agent=agent,
        output_file=output_file
    )

def create_risk_assessor_task(investment_id, investment_name, investment_overview, agent, personal_info, output_file, other_expert_tasks):
    """Creates a Task object for the Risk Assessor."""
    return Task(
        description=f"""        
        # Goal
        You are a risk management expert specializing in EB-5 investments.
        You have been hired by two investors to thoroughly evaluate an EB-5 investment
        for potential risks and red flags.
            - **Investment ID:** {investment_id}
            - **Investment Name:** {investment_name}

        # Introductory Information
        ## Client Information
        Here's some personal background about the two investors that hired you: {personal_info}
        Feel free to ask any other questions that may assist your assessment.

        ## Investment Overview
        Here's an investment overview and a corpus of docs provided by the fund manager. Pay close attention
        to this broad overview because you'll also be able to search these docs for further information.

        Here's the investment overview: {investment_overview}

        # Research Tools
        You have access to the following tools:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - **Knowledge Search**: Access a risk assessment knowledge base for EB-5 investments.
        - **Web Search**: Research industry trends, market risks, and developer/sponsor reputation.
        - **Web Scraper**: Allows you to scrape content from a website.
        
        # Instructions
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the "Search All Documents" and "Search Specific Document" tools.** 
            Look for any red flags related to the financial projections, capital structure, market conditions, developer experience, 
            legal compliance, or exit strategy.
        2. Begin by using "Knowledge Search" to review common EB-5 investment risks.
        3. Thoroughly review other experts' investment analysis to identify & calibrate risks, inconsistencies & red flags, including:
           - Financial: Unrealistic projections, weak capital structure, insufficient collateral, risky exit strategy, etc.
           - Legal: Compliance issues, reputational risks, job creation viability, TEA compliance, policy changes, etc.
           - Developer/Sponsor:  Lack of experience, poor track record, financial instability, conflict of interest
           - Market: Sector volatility, competition, demand or geographical risk, industry-specific exposures, etc.
        4. Use "Web Search" to gather additional information about the sector, market trends, and the developer/sponsor's reputation.
        5. Use "Web Scraper" to scrape content from a website returned from a web search.

        # Deliverable
        - Deliver a **comprehensive risk assessment, highlighting key areas of concern, potential mitigation strategies, and an overall risk rating for the investment.**
            - Please do NOT include general risks not unique to this investment. Base opinions on hard facts & insights.
            - It is absolutely critical to **cite your sources** specific analysis from experts or your own research.
        - Please pay close attention to conflicting pieces of information or insight across expert analysis.
            - When citing conflicts between analysis or experts, please cite sources very specifically.
            - Please include any follow-up questions to ask experts to align on such conflicts.
            - Please do not include follow-up's that are not unique to this investment or don't require other expert opinions.
        - Finally, please provide a thorough & clear summary of insights & risks    
            - Please anchor your opinions on cold hard insights and findings based on robust calculations.
            - Should consider presenting high-level stats, tradeoffs & insights in a table for easy reference.
        - Finally, provide a **conclusion** of the risk assessment, including an overall risk rating for the investment.
        """,
        expected_output="""The risk assessment must include:
        1) A clear identification of all major risks, categorized by type (e.g., financial, legal, market, developer/sponsor).
            - Please do NOT include any general risks not unique to the investment. Base opinions on hard facts & insights.
            - MUST cite sources!
            - Please flag any contradictions between sources or expert analysis.
        2) An evaluation of the severity of each risk, considering likelihood and potential impact.
            - Evaluations MUST cite sources as well. It's important to understand your rationale.
            - Feel free to include follow-up questions for the experts.
        3) Proposed any mitigation strategies for the most significant risks.
        4) An overall risk rating for the investment (e.g., low, medium, high).
        """,
        agent=agent,
        output_file=output_file,
        context=other_expert_tasks
    )

def create_eb5_program_specialist_task(investment_id, investment_name, investment_overview, agent, personal_info, output_file):
    """Creates a Task object for the EB-5 Program Specialist."""
    return Task(
        description=f"""
        # Goal
        You are an expert in the EB-5 program, well-versed in its nuances and requirements.
        You have been hired by two investors to thoroughly evaluate an EB-5 investment, its
        alignment with the program and its risk profile:
            - **Investment ID:** {investment_id}
            - **Investment Name:** {investment_name}
        
        # Introductory Information
        ## Client Information
        Here's some personal background about the two investors that hired you: {personal_info}
        Feel free to ask any other questions that may assist your assessment.

        ## Investment Overview
        Here's an investment overview and a corpus of docs provided by the fund manager. Pay close attention
        to this broad overview because you'll also be able to search these docs for further information.

        Here's the investment overview: {investment_overview}

        # Research Tools
        You have access to the following tools:
        - **Search All Documents:** Allows you to search across all investment documents for relevant information.
        - **Search Specific Document:** Allows you to search within a specific investment document by its name.
        - **Knowledge Search**: Explore a knowledge base containing detailed EB-5 program information.
        - **Web Search**:  Research recent EB-5 program updates, policy changes, and successful project examples.
        - **Web Scraper**: Allows you to scrape content from a website. 

        # Instructions:
        1. **Begin by thoroughly familiarizing yourself with the investment documents using the
           "Search All Documents" and "Search Specific Document" tools.**  Focus on the project structure,
            TEA designation, job creation methodology, and any regional center involvement.
        2. Begin by using "Knowledge Search" to refresh your understanding of EB-5 program requirements, 
            TEA designations, job creation methodologies, EB5 recent reforms, policy updates 
            and past trends on adjudications, fund structures, regional center track records, litigation, etc.
        3. Analyze the investment documents:
           - Project compliance: Does the project's business plan align with EB-5 regulations and the RIA?
                - How is the project addressing the new integrity measures introduced by the RIA?
           - TEA designation & job creation requirements? Designation risks? Litigation vulnerability?
           - Is a regional center involved and if so, what's their track record?  Any past or ongoing regulatory
            issues, lawsuits or complaints? 
           - Fund manager and developer reputational risks? 
                - Important: Fund manager & developer MUST be independent! Lack of independence raises a potential red flag.
                - Any other such reputational risks, conflict of interests, etc.
           - Past precedents, USCIS trends, regulatory changes, etc. that may impact the project.
        4. Use "Web Search" to find comparable projects and understand their success stories or challenges.
        5. Use Web Scraper to get more context abou websites returned by Web Search.

        # Deliverable
        - Provide an **in-depth evaluation of how well this investment aligns with the EB-5 program, 
            highlighting any strengths, weaknesses, or areas requiring further scrutiny.**
        - Please ensure that you **cite your sources** for all important information that you use.
        - Please ensure your analysis is detailed enough to summarize ALL important insights from your research.
        - Please ensure that you **use appropriate formatting** to present your findings.
        - Importantly, please add any follow-up's to consider -- questions for investment team, another expert to consult, etc.
            - Please do NOT include follow-up's that can be researched with provided tools. Please perform such research!
            - Please ensure all follow-up's are about specific information, actionable and not possible with just provided tools.
        """,
        expected_output="""The evaluation should address:
        1) Alignment of the project structure with the EB-5 program's goals.
        2) Justification of the TEA designation & soundness of job creation methodology
        3) Fund Manager reputation & independence? Developer track record?
        4) Reputability and compliance of the regional center (if applicable).
        5) Past precedents, USCIS trends, regulatory changes, etc. that make the project vulernable.
        => Conclusion: Overall strengths, weaknesses, and optionally specific areas needing further attention.
        """,
        agent=agent,
        output_file=output_file
    )