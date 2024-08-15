from transformers import AutoTokenizer, PreTrainedTokenizerFast


class Model:
    tokenizer: PreTrainedTokenizerFast

    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)


Hermes2Pro = Model(model_id="NousResearch/Hermes-2-Pro-Llama-3-8B")
