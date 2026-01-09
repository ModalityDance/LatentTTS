from transformers import GPT2LMHeadModel, GPT2Config


class COCONUTGPT2Config(GPT2Config):
    def __init__(
        self,
        latent_id: int = -100,
        latent_start_id: int = -100,
        latent_end_id: int = -100,
        target_id: int = -100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_id = latent_id
        self.latent_start_id = latent_start_id
        self.latent_end_id = latent_end_id
        self.target_id = target_id


class COCONUTGPT2(GPT2LMHeadModel):
    config_class = COCONUTGPT2Config

    def __init__(self, config):
        super().__init__(config)
