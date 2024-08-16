import typing

from huggingface_hub import login
from pydantic import BaseModel

from libertai_agents.models.base import Model
from libertai_agents.models.hermes import HermesModel
from libertai_agents.models.mistral import MistralModel


class ModelConfiguration(BaseModel):
    vm_url: str
    constructor: typing.Type[Model]


ModelId = typing.Literal[
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "mistralai/Mistral-Nemo-Instruct-2407"
]
MODEL_IDS: list[ModelId] = list(typing.get_args(ModelId))

MODELS_CONFIG: dict[ModelId, ModelConfiguration] = {
    "NousResearch/Hermes-2-Pro-Llama-3-8B": ModelConfiguration(
        vm_url="https://curated.aleph.cloud/vm/84df52ac4466d121ef3bb409bb14f315de7be4ce600e8948d71df6485aa5bcc3/completion",
        constructor=HermesModel),
    "NousResearch/Hermes-3-Llama-3.1-8B": ModelConfiguration(vm_url="http://localhost:8080/completion",
                                                             constructor=HermesModel),
    "mistralai/Mistral-Nemo-Instruct-2407": ModelConfiguration(vm_url="http://localhost:8080/completion",
                                                               constructor=MistralModel)
}


def get_model(model_id: ModelId, hf_token: str | None = None) -> Model:
    model_configuration = MODELS_CONFIG.get(model_id)

    if model_configuration is None:
        raise ValueError(f'model_id must be one of {MODEL_IDS}')

    if hf_token is not None:
        login(hf_token)

    return model_configuration.constructor(model_id=model_id, **model_configuration.model_dump(exclude={'constructor'}))
