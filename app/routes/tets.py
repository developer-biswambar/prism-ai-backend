from azure.identity import ClientCertificateCredential
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# ----------------------------
# 1. Get token using cert auth
# ----------------------------
TENANT_ID = "<your-tenant-id>"
CLIENT_ID = "<your-client-id>"
CERT_PATH = "/path/to/cert.pem"
PRIVATE_KEY_PATH = "/path/to/private.key"
RESOURCE = "https://cognitiveservices.azure.com/"

credential = ClientCertificateCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    certificate_path=CERT_PATH,
    private_key_path=PRIVATE_KEY_PATH,
)

token = credential.get_token(RESOURCE + "/.default")

# ----------------------------
# 2. Configure AzureChatOpenAI
# ----------------------------
self.llm = AzureChatOpenAI(
    azure_endpoint="https://<your-resource-name>.openai.azure.com/",
    deployment_name="<your-deployment-name>",  # e.g., gpt-4o, gpt-4.1-nano
    api_version="2024-08-01-preview",
    openai_api_key=token.token,   # <-- pass bearer token here
    temperature=0,
)

# ----------------------------
# 3. Create SQL Agent (same)
# ----------------------------
self.sql_agent = create_sql_agent(
    llm=self.llm,
    db=self.db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=5,
    handle_parsing_errors=True,
)
