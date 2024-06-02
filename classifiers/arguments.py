from dataclasses import dataclass, field


@dataclass
class ParaphraseQueryFreeArguments:
    files_root_path: str = field(
        default="/home/d.balobin/LLM-Detector-Robustness/data/",
        metadata={"help": "Path to dataset's files of SemEval2024-8."},
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for inference."},
    )
    vllm_model: str = field(
        default="/storage/d.balobin/models/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Name AND path of served vllm model."},
    )
    vllm_port: int = field(
        default=3228,
        metadata={"help": "Port for vllm served model."},
    )
    system_prompt: str = field(
        default="You are a very good rewriter. Rewrite only some words in the provided text, but don't change sense of text. Write only rewrited result.",
        metadata={"help": "System propmt for served model for paraphrasing."},
    )
    result_file_path: str = field(
        default='/home/d.balobin/LLM-Detector-Robustness/test_results.json',
        metadata={"help": "Path where to save intermediate results."},
    )


@dataclass
class ParaphraseQueryBasedArguments(ParaphraseQueryFreeArguments):
    num_generations: int = field(
        default=2,
        metadata={"help": "Number of generations for each example."},
    )
    k: int = field(
        default=3,
        metadata={"help": "Each generation capacity."},
    )
    detector_path: str = field(
        default="/home/d.balobin/LLM-Detector-Robustness/classifiers/model_bpe-llama2.pickle",
        metadata={"help": "Path to sklearn classifier."},
    )
    detector_tokenizer: str = field(
        default="AlexWortega/LLama2-7b",
        metadata={"help": "Path to detector tokenizer."},
    )
    result_file_path: str = field(
        default='/home/d.balobin/LLM-Detector-Robustness/test_results_qb.json',
        metadata={"help": "Path where to save intermediate results."},
    )


@dataclass
class OptimizePropmtArguments(ParaphraseQueryBasedArguments):
    meta_prompt: str = field(
        default="Rewrite the prompt (which i provide you in the next message) for a large language model to rephrase any text so that writes most like a human. You can add some text or change it, but make it small in volume. Just write a new prompt, it must work with any text. Do not ask me about text examples.",
        metadata={"help": "System propmt for paraphrasing paraphrase prompt."},
    )
