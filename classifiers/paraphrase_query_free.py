import re
import json
from typing import cast

from tqdm.auto import tqdm
from langchain_community.llms import VLLMOpenAI
from transformers import AutoTokenizer, HfArgumentParser

from utils import batchify
from load_data import semeval
from arguments import ParaphraseQueryFreeArguments


def main(args: ParaphraseQueryFreeArguments):
    tokenizer = AutoTokenizer.from_pretrained(args.vllm_model)

    ds = semeval(args.files_root_path)
    ds = ds.filter(lambda x: x["label"] == 1)

    records = ds["test"].to_pandas().to_records()
    results = []
    for batch in tqdm(batchify(records, args.batch_size)):
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{args.vllm_port}/v1",
            model_name=args.vllm_model,
            batch_size=args.batch_size,
            top_p=0.9,
            temperature=0.7,
            verbose=False,
            request_timeout=300,
            max_tokens=int(
                max(len(tokenizer(_["text"])["input_ids"]) for _ in batch) * 1.05
            ),
        )

        result = llm.batch(
            [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": row["text"]},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for row in batch
            ],
            config={"max_concurrency": None},
        )
        result = [
            re.sub(r"Here is the rewritten text.*:\n", "", res).strip()
            for res in result
        ]
        results.append(result)
        json.dump(
            results,
            open(args.result_file_path, "w"),
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    parser = HfArgumentParser([ParaphraseQueryFreeArguments])
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(ParaphraseQueryFreeArguments, args)
    main(args)
