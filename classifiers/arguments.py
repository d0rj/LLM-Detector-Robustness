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
