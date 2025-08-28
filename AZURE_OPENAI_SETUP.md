# Azure OpenAI Setup Guide

## Dependencies

✅ **No additional dependencies needed!** 

The Azure OpenAI support uses the existing `openai` package (version 1.51.2) which already includes Azure OpenAI functionality. The proxy support uses `httpx` which is already in your requirements.txt.

## Environment Variables

### Required Environment Variables

Set these environment variables to use Azure OpenAI:

```bash
# Provider Selection
LLM_PROVIDER=azure_openai

# Azure OpenAI Credentials
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```

### Optional Environment Variables

```bash
# Model Configuration
AZURE_OPENAI_MODEL=gpt-4                    # Your deployment name (default: gpt-4)
AZURE_OPENAI_API_VERSION=2024-02-01        # API version (default: 2024-02-01)

# Azure Authentication (optional)
AZURE_TENANT_ID=your-azure-tenant-id       # For enterprise environments

# Model Parameters (optional)
AZURE_OPENAI_TEMPERATURE=0.3               # Response creativity (default: 0.3)
AZURE_OPENAI_MAX_TOKENS=2000               # Max response length (default: 2000)

# Proxy Configuration (optional - for corporate networks)
HTTP_PROXY=http://proxy-server:8080         # HTTP proxy
HTTPS_PROXY=http://proxy-server:8080        # HTTPS proxy

# Or Azure-specific proxy settings
AZURE_HTTP_PROXY=http://proxy-server:8080   # Azure-specific HTTP proxy
AZURE_HTTPS_PROXY=http://proxy-server:8080  # Azure-specific HTTPS proxy
```

## Complete .env File Example

Create or update your `.env` file:

```env
# Azure OpenAI Configuration
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=1234567890abcdef1234567890abcdef
AZURE_OPENAI_ENDPOINT=https://my-company-openai.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-4-deployment
AZURE_TENANT_ID=12345678-1234-5678-9012-123456789012

# Optional: Proxy settings for corporate networks
HTTP_PROXY=http://corporate-proxy:8080
HTTPS_PROXY=http://corporate-proxy:8080

# Optional: Fine-tune model behavior
AZURE_OPENAI_TEMPERATURE=0.3
AZURE_OPENAI_MAX_TOKENS=2000
AZURE_OPENAI_API_VERSION=2024-02-01
```

## How to Get Your Azure OpenAI Credentials

### 1. Azure OpenAI API Key
1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to your Azure OpenAI resource
3. Go to "Keys and Endpoint" section
4. Copy either "KEY 1" or "KEY 2"

### 2. Azure OpenAI Endpoint
1. In the same "Keys and Endpoint" section
2. Copy the "Endpoint" URL (e.g., `https://your-resource.openai.azure.com/`)

### 3. Model Deployment Name
1. In your Azure OpenAI resource, go to "Model deployments"
2. Use the "Deployment name" (not the model name)
3. Example: If you deployed "gpt-4" with deployment name "my-gpt4", use `my-gpt4`

### 4. Tenant ID (Optional)
1. In Azure Portal, go to "Azure Active Directory"
2. Copy the "Tenant ID" from the overview page

## Testing Your Configuration

### Option 1: Using Configuration Validation
```python
from app.config.llm_config import LLMConfig

# Validate your configuration
validation = LLMConfig.validate_configuration("azure_openai")
print(f"Is configured: {validation['is_configured']}")

if not validation['is_configured']:
    print(f"Missing required: {validation['missing_required']}")
    print(f"Missing optional: {validation['missing_optional']}")
    print(f"Warnings: {validation['warnings']}")
```

### Option 2: Using the LLM Service Directly
```python
from app.services.llm_service import get_llm_service, LLMMessage

# Get the service (will use Azure OpenAI if configured)
service = get_llm_service()
print(f"Using provider: {service.get_provider_name()}")
print(f"Using model: {service.get_model_name()}")

# Test a simple request
messages = [
    LLMMessage(role="user", content="Hello, can you confirm you're working?")
]

response = service.generate_text(messages)
print(f"Success: {response.success}")
if response.success:
    print(f"Response: {response.content}")
else:
    print(f"Error: {response.error}")
```

### Option 3: Using Programmatic Configuration
```python
from app.services.llm_service import configure_and_set_azure_openai

# Configure Azure OpenAI programmatically (instead of env vars)
configure_and_set_azure_openai(
    api_key="your-api-key",
    endpoint="https://your-resource.openai.azure.com/",
    model="gpt-4",
    tenant_id="your-tenant-id",
    proxy_config={"http": "http://proxy:8080", "https": "http://proxy:8080"}
)

# Now use get_llm_service() as normal
service = get_llm_service()
```

## Troubleshooting

### Common Issues

1. **"Azure OpenAI service not available"**
   - Check that `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are set
   - Verify the API key is correct
   - Ensure the endpoint URL ends with a `/`

2. **"Model deployment not found"**
   - Use the deployment name, not the model name
   - Check your deployment name in Azure Portal → Azure OpenAI → Model deployments

3. **Proxy connection issues**
   - Verify proxy settings are correct
   - Try with and without proxy settings
   - Check if corporate firewall allows Azure OpenAI endpoints

4. **Authentication errors**
   - Verify your Azure subscription has access to Azure OpenAI
   - Check if your tenant has Azure OpenAI enabled
   - Ensure the API key hasn't expired

### Checking Configuration
```bash
# Check if environment variables are loaded
python -c "
from app.config.llm_config import LLMConfig
import json
config = LLMConfig.get_provider_config('azure_openai')
print(json.dumps({k: '***' if 'key' in k.lower() else v for k, v in config.items()}, indent=2))
"
```

### Debug Mode
Set these for more detailed logging:
```bash
export PYTHONPATH=/path/to/your/project
export LOG_LEVEL=DEBUG
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables or Azure Key Vault**
3. **Rotate API keys regularly**
4. **Use least-privilege access policies**
5. **Monitor API usage and costs**
6. **Use HTTPS proxies only**

## Integration with Existing Endpoints

Once configured, Azure OpenAI will work with all existing endpoints:

- ✅ `/transformation/generate-config/` - AI-powered transformation rules
- ✅ `/reconciliation/generate-regex/` - Regex pattern generation  
- ✅ `/delta/generate-config/` - Delta configuration generation
- ✅ All other LLM-powered features

No code changes needed - just set the environment variables!