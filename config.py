# global config for the project

# LLM-related values
MODEL_NAME = "local-llama"
LLAMA_PATH = "~/.ollama/models/manifests/registry.ollama.ai/library/llama3/8b-instruct-q8_0"
TEMPERATURE = 0.75 # TODO: Ensure this is actually honored in calls to the models
MAX_TOKENS = 100000 # TODO: Ensure this is actually honored
TOP_P=0.95 # TODO: Ensure this is actually honored

# Knwowledge-base paths
KB_PATH = "knowledge_bases"
KB_PATH_FINANCIAL_ANALYST = KB_PATH + "/financial_analysis.txt"
KB_PATH_EB5_PROGRAM = KB_PATH + "/eb5_program.txt"
KB_PATH_IMMIGRATION_LAW = KB_PATH + "/immigration_law.txt"
KB_PATH_RISK_ASSESSMENT = KB_PATH + "/risk_assessment.txt"
