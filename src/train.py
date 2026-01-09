from transformers import HfArgumentParser

from src.trainer import LatentRMTrainer, LatentRMConfig
# from trainers.seperate_latent_prm_trainer import LatentPRMTrainer, LatentPRMConfig


def main(*args, **kwargs):
    parser = HfArgumentParser(LatentRMConfig)
    if len(args) == 1:
        if len(kwargs) > 0:
            raise ValueError(f"Invalid arguments: {args} and {kwargs}")
        if "yaml" in args[0]:
            trainer_args = parser.parse_yaml_file(args[0])[0]
        elif "json" in args[0]:
            trainer_args = parser.parse_json_file(args[0])[0]
        else:
            raise ValueError(f"Invalid arguments: {args}, only yaml and json are supported")
    elif len(args) == 0:
        trainer_args = parser.parse_dict(kwargs)[0]
    else:
        raise ValueError(f"Invalid arguments: {args}")

    trainer = LatentRMTrainer(args=trainer_args)
    trainer.train()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
