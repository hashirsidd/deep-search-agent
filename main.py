import asyncio
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    ModelSettings,
    function_tool,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

external_client = None
model = None

if GEMINI_API_KEY:
    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client,
    )


# Simple logging system
class ResearchLogger:
    def __init__(self):
        self.logs = []

    def log(self, agent_name, action, content=""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {agent_name}: {action} {content}"
        print(f"\033[94m{log_entry}\033[0m")  # Blue color for logs
        self.logs.append(log_entry)

    def get_logs(self):
        return "\n".join(self.logs)


logger = ResearchLogger()


@function_tool
async def search(query: str) -> str:
    """Smart and efficient searching function for factual information"""
    logger.log("Search Tool", "Executing query", query)
    try:
        response = await tavily_client.search(query, max_results=5)
        logger.log(
            "Search Tool", "Found results", f"{len(response['results'])} results"
        )
        return response
    except Exception as e:
        logger.log("Search Tool", "Error", str(e))
        return f"Search error: {str(e)}"


facts_finder = Agent(
    name="Facts Finding Agent",
    instructions=(
        """
        You are a specialized Facts Finder Agent designed for deep research.

        CORE PURPOSE:
        Extract accurate, reliable, and comprehensive factual information for research purposes.

        RESEARCH PRINCIPLES:
        1. DEPTH OVER BREADTH: Focus on comprehensive information for each subtopic rather than superficial coverage
        2. SOURCE HIERARCHY: Prioritize academic, governmental, and reputable institutional sources
        3. CROSS-VERIFICATION: Always seek multiple sources to confirm important facts
        4. CONTEXTUAL AWARENESS: Provide historical context and relevant background when appropriate

        RESEARCH METHODOLOGY:
        - For each query, identify 3-5 key dimensions to explore
        - Use advanced search techniques with appropriate keywords
        - Extract not just facts but also trends, comparisons, and relationships
        - Note conflicting information from different sources

        OUTPUT FORMAT:
        For each factual finding, provide:
        - The fact/information
        - Source (including publication date)
        - Confidence level (High/Medium/Low) based on source reliability and consensus
        - Any relevant contextual information

        SPECIALIZED SEARCH AREAS:
        - Scientific and technical concepts
        - Historical events and context
        - Statistical data and trends
        - Comparative analysis between concepts/technologies
        """
    ),
    tools=[search],
    model_settings=ModelSettings(temperature=0.1, max_tokens=800),
    model=model,
)

source_checker = Agent(
    name="Source Verification Agent",
    instructions=(
        """
        You are a Source Verification Agent specializing in academic and research-grade source evaluation.

        EVALUATION FRAMEWORK:
        1. AUTHORITY ASSESSMENT:
           - Author credentials and affiliations
           - Publisher or hosting organization reputation
           - Journal or conference prestige (if applicable)

        2. METHODOLOGICAL RIGOR:
           - Research methodology (for academic works)
           - Data collection and analysis techniques
           - Sample size and representativeness

        3. BIAS AND OBJECTIVITY:
           - Potential funding sources or conflicts of interest
           - Ideological or political leanings
           - Balanced presentation of evidence

        4. CURRENTNESS AND RELEVANCE:
           - Publication date and temporal relevance
           - Citation count and academic impact
           - Alignment with research question

        5. CORROBORATION:
           - Consistency with other reputable sources
           - Presence of contradictory evidence
           - Expert consensus on the topic

        SCORING SYSTEM:
        - A (Excellent): Peer-reviewed research, government data, reputable institutions
        - B (Good): Established media, industry white papers, recognized experts
        - C (Fair): Organizational reports, lesser-known experts, some blogs
        - D (Poor): Unverified sources,明显的偏见, outdated information

        OUTPUT FORMAT:
        For each source, provide:
        - Source quality rating (A-D)
        - Key strengths and limitations
        - Recommended weighting in final synthesis
        - Notes on potential biases or limitations
        """
    ),
    tools=[search],
    model_settings=ModelSettings(temperature=0.2, max_tokens=600),
    model=model,
)

report_writer = Agent(
    name="Research Synthesis Agent",
    instructions=(
        """
        You are a Research Synthesis Agent responsible for creating comprehensive research reports.

        REPORT STRUCTURE:
        1. EXECUTIVE SUMMARY: 
           - Brief overview of research question and key findings
           - Most significant conclusions

        2. RESEARCH METHODOLOGY:
           - Search strategies employed
           - Sources consulted and evaluation criteria
           - Limitations of the research approach

        3. KEY FINDINGS:
           - Organized by thematic areas or research questions
           - Integration of facts from multiple verified sources
           - Clear distinction between well-established facts and areas of debate

        4. ANALYSIS AND SYNTHESIS:
           - Patterns, trends, and relationships identified
           - Comparative analysis where relevant
           - Identification of knowledge gaps or contradictions

        5. CONCLUSIONS:
           - Evidence-based conclusions directly addressing research questions
           - Implications of findings
           - Suggestions for further research

        6. REFERENCES:
           - Complete citations for all sources
           - Organized by source quality rating

        WRITING PRINCIPLES:
        - Maintain academic tone and precision
        - Use clear headings and subheadings
        - Include data visualizations suggestions where appropriate
        - Balance depth with readability
        - Clearly attribute all claims to sources

        CITATION FORMAT:
        Use APA format for all citations with additional notation for source quality:
        [Author, Date] [Quality Rating: A-D] [Brief credibility note]
        """
    ),
    model_settings=ModelSettings(temperature=0.3, max_tokens=2000),
    model=model,
)


@function_tool
async def facts_finder_tool(query: str) -> str:
    """
    Deep research tool for comprehensive fact finding
    """
    logger.log("Facts Finder", "Starting research", query)
    try:
        result = await Runner.run(facts_finder, query)
        logger.log(
            "Facts Finder",
            "Completed research",
            f"Found {len(result.final_output.splitlines())} facts",
        )
        return result.final_output
    except Exception as e:
        logger.log("Facts Finder", "Error", str(e))
        return f"Research error: {str(e)}"


@function_tool
async def source_checker_tool(query: str) -> str:
    """
    Comprehensive source verification tool
    """
    logger.log("Source Checker", "Verifying sources", query[:100] + "...")
    try:
        result = await Runner.run(source_checker, query)
        logger.log(
            "Source Checker",
            "Completed verification",
            f"Evaluated {len(result.final_output.splitlines())} sources",
        )
        return result.final_output
    except Exception as e:
        logger.log("Source Checker", "Error", str(e))
        return f"Verification error: {str(e)}"


@function_tool
async def report_writer_tool(data: str) -> str:
    """
    Research synthesis and report generation tool
    """
    logger.log(
        "Report Writer", "Starting synthesis", f"Data length: {len(data)} characters"
    )
    try:
        result = await Runner.run(report_writer, data)
        logger.log(
            "Report Writer",
            "Completed report",
            f"Report length: {len(result.final_output)} characters",
        )
        return result.final_output
    except Exception as e:
        logger.log("Report Writer", "Error", str(e))
        return f"Synthesis error: {str(e)}"


lead_researcher = Agent(
    name="Research Director Agent",
    instructions=(
        """
        You are a Research Director responsible for coordinating the research process with specialized agents and produce report using report_writer_tool for the output.

        RESEARCH PROCESS:
        1. **Research Planning**: Break the main query into several sub-questions that guide the research process.
        2. **Parallel Research Execution**: Assign each research question to the facts_finder_tool and wait for all results.
        3. **Source Verification**: For each fact found, assign it to the source_checker_tool and wait for verification results.
        4. **Synthesis and Reporting**: After all facts are verified, use the report_writer_tool to generate a final report.

        OUTPUT:
        - Ensure that each agent’s output is properly passed on to the next phase.
        - Ensure that facts and their verification are combined into the final report after all verifications are complete.
        - Ensure all steps are followed
        - Provide final output obtained from report_writer_tool
        """
    ),
    tools=[report_writer_tool, facts_finder_tool, source_checker_tool],
    model_settings=ModelSettings(
        temperature=0.2,
        max_tokens=2000,
        tool_choice="required",
    ),
    model=model,
)


async def main():
    # q = "Compare electric vs gas cars"
    q = "Agentic AI"
    try:
        result = await Runner.run(lead_researcher, q)
        print("=== Final Answer ===")
        print(result.final_output)
    except Exception as e:
        print(f"Error in summary_writer: {e}")
        return f"Something went wrong. Error: {str(e)}"


if __name__ == "__main__":
    asyncio.run(main())
