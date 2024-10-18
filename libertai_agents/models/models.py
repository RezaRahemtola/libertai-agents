import typing

from huggingface_hub import login
from pydantic import BaseModel

from libertai_agents.models.base import Model, ModelId
from libertai_agents.models.hermes import HermesModel
from libertai_agents.models.mistral import MistralModel


class ModelConfiguration(BaseModel):
    vm_url: str
    context_length: int
    constructor: typing.Type[Model]


MODEL_IDS: list[ModelId] = list(typing.get_args(ModelId))

# TODO: update URLs with prod, and check context size (if we deploy it with a lower one)
MODELS_CONFIG: dict[ModelId, ModelConfiguration] = {
    "NousResearch/Hermes-2-Pro-Llama-3-8B": ModelConfiguration(
        vm_url="https://curated.aleph.cloud/vm/84df52ac4466d121ef3bb409bb14f315de7be4ce600e8948d71df6485aa5bcc3/completion",
        context_length=8192,
        constructor=HermesModel),
    "NousResearch/Hermes-3-Llama-3.1-8B": ModelConfiguration(vm_url="http://localhost:8080/completion",
                                                             context_length=131_072,
                                                             constructor=HermesModel),
    "mistralai/Mistral-Nemo-Instruct-2407": ModelConfiguration(vm_url="http://localhost:8080/completion",
                                                               context_length=131_072,
                                                               constructor=MistralModel)
}


def get_model(model_id: ModelId, hf_token: str | None = None) -> Model:
    """
    Get one of the available models

    :param model_id: HuggingFace ID of the model, must be one of the supported models
    :param hf_token: Optional access token, required to use gated models
    :return: An instance of the model
    """
    model_configuration = MODELS_CONFIG.get(model_id)

    if model_configuration is None:
        raise ValueError(f'model_id must be one of {MODEL_IDS}')

    if hf_token is not None:
        login(hf_token)

    return model_configuration.constructor(model_id=model_id, **model_configuration.dict(exclude={'constructor'}))
