# Configuration Guide

This guide explains how to configure GDPR-Bench-Android for different use cases.

## API Key Configuration

### Quick Setup

The simplest way to get started is to set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

This will work for all OpenAI-compatible models (GPT-4o, O1, etc.).

### Detailed Configuration

#### Option 1: Environment Variables (Recommended)

Set environment variables before running the scripts:

**Linux/Mac:**
```bash
export OPENAI_API_KEY='sk-your-openai-key'
export OPENAI_API_BASE='https://api.openai.com/v1'
export ANTHROPIC_API_KEY='sk-your-anthropic-key'
export GOOGLE_API_KEY='your-google-key'
```

**Windows Command Prompt:**
```cmd
set OPENAI_API_KEY=sk-your-openai-key
set OPENAI_API_BASE=https://api.openai.com/v1
```

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY='sk-your-openai-key'
$env:OPENAI_API_BASE='https://api.openai.com/v1'
```

#### Option 2: .env File

1. Copy the example file:
   ```bash
   cp config.env.example .env
   ```

2. Edit `.env` and add your keys:
   ```bash
   OPENAI_API_KEY=sk-your-actual-key-here
   OPENAI_API_BASE=https://api.openai.com/v1
   ```

3. Load the environment variables:
   ```bash
   # Linux/Mac
   source .env
   
   # Or use python-dotenv
   pip install python-dotenv
   ```

4. If using python-dotenv, add this to your script:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Load .env file
   ```

#### Option 3: Custom API Endpoint

If you're using a proxy or custom API endpoint (like Azure OpenAI):

```bash
export OPENAI_API_BASE='https://your-custom-endpoint.com/v1'
export OPENAI_API_KEY='your-key-for-custom-endpoint'
```

## Method-Specific Configuration

### ReAct Agent

The ReAct agent reads configuration from `methods/config.py`:

```python
METHOD_CONFIGS = {
    'react': {
        'model': 'gpt-4o',
        'api_base': os.environ.get('OPENAI_API_BASE'),
        'api_key': os.environ.get('OPENAI_API_KEY'),
        'max_iterations': 5,
        'temperature': 0.0,
        'timeout': 300,
    }
}
```

Customize by setting environment variables before running.

### RAG Method

For RAG-based detection, you can use separate keys for embeddings:

```bash
export OPENAI_API_KEY='your-llm-key'
export OPENAI_EMBEDDING_API_KEY='your-embedding-key'
```

If not set, it will use `OPENAI_API_KEY` for both.

### AST-based Method

No API key needed - this method uses local static analysis.

## Model-Specific Keys

If you want to use different API keys for different model providers:

```bash
# OpenAI models (GPT-4o, O1)
export OPENAI_API_KEY='sk-your-openai-key'

# Anthropic models (Claude)
export ANTHROPIC_API_KEY='sk-ant-your-anthropic-key'

# Google models (Gemini)
export GOOGLE_API_KEY='your-google-api-key'
```

**Note:** Currently, `predict.py` uses a single API key for all models. To use different keys, you need to modify the `model_to_key_task1` and `model_to_key_task2` dictionaries in `predict.py`.

## Security Best Practices

1. **Never commit API keys** to git
   - `.env` files are automatically ignored
   - Double-check before committing

2. **Use environment variables** in production
   - Don't hardcode keys in source code
   - Use secure secret management systems

3. **Rotate keys regularly**
   - Generate new keys periodically
   - Revoke old keys

4. **Limit key permissions**
   - Use API keys with minimal required permissions
   - Set usage limits if available

## Troubleshooting

### "No API key found"

**Problem:** You see: `⚠️ Warning: No API key found in environment variables.`

**Solution:**
```bash
# Check if the key is set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY='your-key-here'
```

### "API call failed"

**Problem:** API calls are failing or timing out.

**Solutions:**
1. Check your API key is valid
2. Verify your API endpoint URL
3. Check your internet connection
4. Verify you have sufficient API credits

### "Invalid API base URL"

**Problem:** API calls fail with URL-related errors.

**Solution:**
```bash
# Make sure the base URL doesn't end with /chat/completions
# predict.py will add that automatically
export OPENAI_API_BASE='https://api.openai.com/v1'  # Correct
# NOT: https://api.openai.com/v1/chat/completions  # Wrong
```

## Testing Your Configuration

Quick test to verify your API key works:

```bash
# Set your key
export OPENAI_API_KEY='your-key-here'

# Run a quick prediction on AST-based method (no API needed)
python predict.py --models=ast-based --tasks=2 --exclude-apps=Android_Spy_App

# If successful, try with an LLM model
python predict.py --models=gpt-4o --tasks=2 --exclude-apps=Android_Spy_App
```

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Google AI Studio](https://ai.google.dev/)

