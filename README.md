# LibertAI Agents

## Supported models

We support multiple open-source models that have agentic capabilities.

- [Hermes 2 Pro - Llama 3 8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
- ⏳ [Hermes 3 - Llama-3.1 8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
- ⏳ [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)

## Using a gated model

Some models, like [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) are gated (
generally to require you to accept some usage conditions).\
To use those models, you need to create an [access token](https://huggingface.co/settings/tokens) from your Hugging Face
account and give it to the `get_model` function.