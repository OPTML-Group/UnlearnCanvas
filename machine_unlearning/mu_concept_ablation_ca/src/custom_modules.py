from packaging import version
import torch.nn as nn
import transformers
from transformers import CLIPTokenizer, CLIPTextModel


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedderWrapper(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, modifier_token, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.modifier_token = modifier_token
        self.modifier_token_id = self.tokenizer(self.modifier_token)['input_ids'][1:-1]
        print(self.modifier_token, self.modifier_token_id)

        self.freeze()

    def custom_forward(self, hidden_states, input_ids):
        r"""
        Returns:
        """
        input_shape = hidden_states.size()
        bsz, seq_len = input_shape[:2]
        if version.parse(transformers.__version__) >= version.parse('4.21'):
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )
        else:
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len).to(
                hidden_states.device
            )

        encoder_outputs = self.transformer.text_model.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)

        return last_hidden_state

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.text_model.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.final_layer_norm.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.embeddings.position_embedding.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        indices = tokens == self.modifier_token_id[-1]
        for token_id in self.modifier_token_id:
            indices |= tokens == token_id

        indices = (indices*1).unsqueeze(-1)

        input_shape = tokens.size()
        tokens = tokens.view(-1, input_shape[-1])

        hidden_states = self.transformer.text_model.embeddings(input_ids=tokens)
        hidden_states = (1-indices)*hidden_states.detach() + indices*hidden_states

        z = self.custom_forward(hidden_states, tokens)

        return z

    def encode(self, text):
        return self(text)


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedderWrapper()
    count_params(model, verbose=True)
