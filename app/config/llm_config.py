# Simplified LLM Configuration
"""
Configuration settings for LLM providers.
Supports OpenAI, Azure OpenAI, and JPMC LLM services.
"""

import os
from typing import Dict, Any


class LLMConfig:
    """Configuration class for LLM providers"""

    # Default provider settings
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODELS = {
        "openai": "gpt-4",
        "azure_openai": "gpt-4",
        "jpmcllm": "jpmc-llm-v1",
    }

    @classmethod
    def get_provider(cls) -> str:
        """Get the configured LLM provider"""
        return os.getenv('LLM_PROVIDER', cls.DEFAULT_PROVIDER).lower()

    @classmethod
    def get_model(cls, provider: str = None) -> str:
        """Get the configured model for a provider"""
        provider = provider or cls.get_provider()

        # Check for provider-specific model override
        model_env_var = f'{provider.upper()}_MODEL'
        if os.getenv(model_env_var):
            return os.getenv(model_env_var)

        # Check for general model override
        if os.getenv('LLM_MODEL'):
            return os.getenv('LLM_MODEL')

        # Fall back to default
        return cls.DEFAULT_MODELS.get(provider, cls.DEFAULT_MODELS[cls.DEFAULT_PROVIDER])

    @classmethod
    def get_provider_config(cls, provider: str = None) -> Dict[str, Any]:
        """Get full configuration for a provider"""
        provider = provider or cls.get_provider()

        config = {
            "provider": provider,
            "model": cls.get_model(provider),
        }

        # Add provider-specific settings
        if provider == "openai":
            config.update({
                "api_key": os.getenv('OPENAI_API_KEY'),
                "temperature": float(os.getenv('OPENAI_TEMPERATURE', '0.3')),
                "max_tokens": int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
            })
        elif provider == "azure_openai":
            config.update({
                "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
                "endpoint": os.getenv('AZURE_OPENAI_ENDPOINT'),
                "api_version": os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
                "tenant_id": os.getenv('AZURE_TENANT_ID'),
                "temperature": float(os.getenv('AZURE_OPENAI_TEMPERATURE', '0.3')),
                "max_tokens": int(os.getenv('AZURE_OPENAI_MAX_TOKENS', '2000')),
                # Parse proxy configuration from environment variables
                "proxy": cls._get_proxy_config()
            })
        elif provider == "jpmcllm":
            config.update({
                "api_url": os.getenv('JPMC_LLM_URL'),
                "temperature": float(os.getenv('JPMC_LLM_TEMPERATURE', '0.3')),
                "max_tokens": int(os.getenv('JPMC_LLM_MAX_TOKENS', '2000')),
                "timeout": int(os.getenv('JPMC_LLM_TIMEOUT', '30')),
            })

        return config

    @classmethod
    def _get_proxy_config(cls) -> Dict[str, str]:
        """Get proxy configuration from environment variables"""
        proxy_config = {}
        
        # Check for HTTP proxy
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        if http_proxy:
            proxy_config['http'] = http_proxy
        
        # Check for HTTPS proxy
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        if https_proxy:
            proxy_config['https'] = https_proxy
        
        # Check for Azure-specific proxy settings
        azure_http_proxy = os.getenv('AZURE_HTTP_PROXY')
        azure_https_proxy = os.getenv('AZURE_HTTPS_PROXY')
        
        if azure_http_proxy:
            proxy_config['http'] = azure_http_proxy
        if azure_https_proxy:
            proxy_config['https'] = azure_https_proxy
        
        return proxy_config if proxy_config else None

    @classmethod
    def is_provider_configured(cls, provider: str) -> bool:
        """Check if a provider is properly configured"""
        config = cls.get_provider_config(provider)

        if provider == "openai":
            return bool(config.get("api_key"))
        elif provider == "azure_openai":
            return bool(config.get("api_key") and config.get("endpoint"))
        elif provider == "jpmcllm":
            return bool(config.get("api_url"))

        return False

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of configured providers"""
        available = []
        for provider in cls.DEFAULT_MODELS.keys():
            if cls.is_provider_configured(provider):
                available.append(provider)
        return available

    @classmethod
    def configure_azure_openai_env(
        cls, 
        api_key: str, 
        endpoint: str, 
        model: str = "gpt-4",
        api_version: str = "2024-02-01",
        tenant_id: str = None
    ):
        """
        Configure Azure OpenAI using environment variables.
        This is a helper method to set the environment variables programmatically.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            model: Model deployment name (default: "gpt-4")
            api_version: Azure OpenAI API version (default: "2024-02-01")
            tenant_id: Azure tenant ID (optional)
        """
        os.environ['LLM_PROVIDER'] = 'azure_openai'
        os.environ['AZURE_OPENAI_API_KEY'] = api_key
        os.environ['AZURE_OPENAI_ENDPOINT'] = endpoint
        os.environ['AZURE_OPENAI_MODEL'] = model
        os.environ['AZURE_OPENAI_API_VERSION'] = api_version
        
        if tenant_id:
            os.environ['AZURE_TENANT_ID'] = tenant_id

    @classmethod
    def get_configuration_help(cls) -> str:
        """Get help text for configuring LLM providers"""
        help_text = """
LLM Provider Configuration Help:

1. OpenAI:
   Environment Variables:
   - LLM_PROVIDER=openai
   - OPENAI_API_KEY=your-api-key
   - OPENAI_MODEL=gpt-4 (optional)
   - OPENAI_TEMPERATURE=0.3 (optional)
   - OPENAI_MAX_TOKENS=2000 (optional)

2. Azure OpenAI:
   Environment Variables:
   - LLM_PROVIDER=azure_openai
   - AZURE_OPENAI_API_KEY=your-azure-api-key
   - AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   - AZURE_OPENAI_MODEL=your-deployment-name (optional, default: gpt-4)
   - AZURE_OPENAI_API_VERSION=2024-02-01 (optional)
   - AZURE_TENANT_ID=your-tenant-id (optional)
   - AZURE_OPENAI_TEMPERATURE=0.3 (optional)
   - AZURE_OPENAI_MAX_TOKENS=2000 (optional)
   
   Proxy Settings (optional):
   - HTTP_PROXY=http://proxy:8080
   - HTTPS_PROXY=http://proxy:8080
   - AZURE_HTTP_PROXY=http://proxy:8080 (Azure-specific)
   - AZURE_HTTPS_PROXY=http://proxy:8080 (Azure-specific)

3. JPMC LLM:
   Environment Variables:
   - LLM_PROVIDER=jpmcllm
   - JPMC_LLM_URL=http://your-jpmc-llm-endpoint
   - JPMC_LLM_MODEL=jpmc-llm-v1 (optional)
   - JPMC_LLM_TEMPERATURE=0.3 (optional)
   - JPMC_LLM_MAX_TOKENS=2000 (optional)
   - JPMC_LLM_TIMEOUT=30 (optional)

Example .env file:
```
# Use Azure OpenAI
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-4
AZURE_TENANT_ID=your-tenant-id
```
"""
        return help_text.strip()

    @classmethod
    def validate_configuration(cls, provider: str = None) -> Dict[str, Any]:
        """
        Validate the configuration for a provider and return detailed status.
        
        Args:
            provider: Provider to validate (default: current provider)
            
        Returns:
            Dict with validation results
        """
        provider = provider or cls.get_provider()
        config = cls.get_provider_config(provider)
        
        result = {
            "provider": provider,
            "is_configured": cls.is_provider_configured(provider),
            "missing_required": [],
            "missing_optional": [],
            "warnings": [],
            "config": config
        }
        
        # Check required fields based on provider
        if provider == "openai":
            if not config.get("api_key"):
                result["missing_required"].append("OPENAI_API_KEY")
                
        elif provider == "azure_openai":
            if not config.get("api_key"):
                result["missing_required"].append("AZURE_OPENAI_API_KEY")
            if not config.get("endpoint"):
                result["missing_required"].append("AZURE_OPENAI_ENDPOINT")
            
            # Check optional but recommended
            if not config.get("tenant_id"):
                result["missing_optional"].append("AZURE_TENANT_ID")
            if not config.get("proxy") and (os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")):
                result["warnings"].append("HTTP/HTTPS proxy detected but not configured for Azure OpenAI")
                
        elif provider == "jpmcllm":
            if not config.get("api_url"):
                result["missing_required"].append("JPMC_LLM_URL")
        
        return result
