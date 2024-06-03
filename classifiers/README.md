# LLM-Detector-Robustness / classifiers

Исследование применимости атак на детекторы сгенерированного LLM текстов.

## Usage

### Run local Llama-3-8B model

Я использовал [vllm](https://github.com/vllm-project/vllm), можно использовать другие методы разворачивания. Главное, чтобы API было openai-like, чтобы код от langchain можно было легко изменить.

```bash
python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --port 3228
```

### Run selected dataset paraphrasing

Для получения списка доступных аргументов для каждого скрипта можно воспользоваться командой `--help`, или посмотреть агрументы в файле [`arguments.py`](./arguments.py):

```bash
python paraphrase_query_free.py --help
```

```bash
python paraphrase_query_based.py --help
```

Из важных агрументов:

- `--files_root_path` - путь до train/dev/test файлов соревнования semeval2024 8 subtask A monolingual. Их можно скачать [с официальной ссылки](https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc);
- `--detector_path` - путь до sklearn pipeline в формате pickle, который выдаёт score от 0 до 1, где 1 - текст сгененерирован LLM;
- `--detector_tokenizer` - путь до `transformers` токенизатора для детектора (см. `--detector_path`);

### Run prompt optimization via genetic algorithm

```bash
python optimize_prompt.py --help
```
