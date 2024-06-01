import re
import json
import pickle
from typing import cast

import pygad
import numpy as np
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from langchain_community.llms import VLLMOpenAI
from transformers import AutoTokenizer, HfArgumentParser

from utils import batchify
from load_data import semeval
from arguments import ParaphraseQueryBasedArguments


class TextGenerationGA:
    def __init__(
                self,
                tokenizer: AutoTokenizer,
                detector: BaseEstimator,
                detector_tokenizer: AutoTokenizer,
                args: ParaphraseQueryBasedArguments,
        ):
        self.tokenizer = tokenizer
        self.detector = detector
        self.detector_tokenizer = detector_tokenizer
        self.args = args
        self.current_paraphrases = []

    def generate_paraphrases(self, text: str, k: int) -> list:
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{args.vllm_port}/v1",
            model_name=args.vllm_model,
            batch_size=args.batch_size,
            top_p=0.9,
            temperature=0.7,
            verbose=False,
            request_timeout=300,
            max_tokens=int(len(self.tokenizer(text)["input_ids"]) * 1.05),
        )
        result = llm.batch(
            [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.args.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            ] * k,
            config={"max_concurrency": None},
        )
        paraphrases = [
            re.sub(r"Here is the rewritten text.*:\n", "", res).strip()
            for res in result
        ]
        return paraphrases

    def fitness_func(self, ga_instance, solution, solution_idx):
        paraphrase = self.current_paraphrases[int(solution[0])]
        prediction = self.detector.predict([paraphrase])
        return 1 - prediction[0]

    def on_generation(self, ga_instance):
        best_solution = ga_instance.best_solution()
        best_paraphrase = self.current_paraphrases[int(best_solution[0])]
        self.current_paraphrases = self.generate_paraphrases(best_paraphrase, len(ga_instance.population))
        ga_instance.population = np.arange(len(self.current_paraphrases)).reshape((len(self.current_paraphrases), 1))

    def run_ga(self, initial_text: str):
        self.current_paraphrases = self.generate_paraphrases(initial_text, self.args.k)

        ga_instance = pygad.GA(
            num_generations=self.args.num_generations,
            num_parents_mating=self.args.k,
            fitness_func=self.fitness_func,
            sol_per_pop=self.args.k,
            num_genes=1,
            initial_population=np.arange(self.args.k).reshape((self.args.k, 1)),
            on_generation=self.on_generation
        )

        ga_instance.run()
        best_solution = ga_instance.best_solution()
        return self.current_paraphrases[int(best_solution[0])]


def main(args: ParaphraseQueryBasedArguments):
    tokenizer = AutoTokenizer.from_pretrained(args.vllm_model)
    detector = pickle.load(open(args.detector_path, "rb"))
    detector_tokenizer = AutoTokenizer.from_pretrained(args.detector_tokenizer)

    ds = semeval(args.files_root_path)
    ds = ds.filter(lambda x: x["label"] == 1)

    records = ds["test"].to_pandas().to_records()
    results = []
    for row in tqdm(records):
        ga_instance = TextGenerationGA(
            tokenizer,
            detector,
            detector_tokenizer,
            args,
        )
        best_paraphrase = ga_instance.run_ga(row["text"])
        results.append(best_paraphrase)

        json.dump(
            results,
            open(args.result_file_path, "w"),
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    parser = HfArgumentParser([ParaphraseQueryBasedArguments])
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(ParaphraseQueryBasedArguments, args)
    main(args)
