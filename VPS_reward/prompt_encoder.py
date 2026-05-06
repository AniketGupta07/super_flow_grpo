"""
Generate SBERT prompt embeddings for VPS reward model training.

Usage:
    python prompt_encoder.py --input data/dataset.json --output data/SBERT_PROMPT_EMBEDDINGS.json
    python prompt_encoder.py --input data/prompts.txt --output data/SBERT_PROMPT_EMBEDDINGS.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
NORMALIZE = True


def load_prompt_items(input_path: str | Path) -> list[dict]:
    """Load items with at least ``prompt`` and ``index`` keys from JSON or plain text.

    Text files: one prompt per line (UTF-8). Empty lines are skipped. Indices are
    1-based in order of non-empty lines, matching the JSON convention used elsewhere.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        items = []
        idx = 1
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                items.append({"index": idx, "prompt": p})
                idx += 1
        return items

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON input must be a list of objects, got {type(data).__name__}")
    return data


class SbertPromptEmbedder:
    def __init__(self, model_name=SBERT_MODEL_NAME, device=DEVICE):
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def embed_prompts(self, prompts, batch_size=BATCH_SIZE, normalize=NORMALIZE):
        # returns torch.Tensor [len(prompts), dim]
        return self.model.encode(
            prompts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SBERT prompt embeddings for VPS reward model."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input file (JSON with 'prompt' field, or .txt with one prompt per line)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output JSON file for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for encoding (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SBERT_MODEL_NAME,
        help=f"SBERT model name (default: {SBERT_MODEL_NAME})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    
    embedder = SbertPromptEmbedder(model_name=args.model)

    data = load_prompt_items(args.input)

    # --- DEDUPE PROMPTS ---
    prompt_to_positions = {}
    for pos, item in enumerate(data):
        p = item["prompt"]
        if p not in prompt_to_positions:
            prompt_to_positions[p] = []
        prompt_to_positions[p].append(pos)

    unique_prompts = list(prompt_to_positions.keys())
    total_unique = len(unique_prompts)
    print(f"Total items: {len(data)} | Unique prompts: {total_unique}")

    # --- EMBED UNIQUE PROMPTS IN BATCHES ---
    prompt_to_embedding = {}
    start_time = time.time()

    for start in range(0, total_unique, args.batch_size):
        end = min(start + args.batch_size, total_unique)
        batch_prompts = unique_prompts[start:end]

        batch_embeds = embedder.embed_prompts(batch_prompts, batch_size=args.batch_size)
        batch_embeds = batch_embeds.detach().cpu()

        for i, p in enumerate(batch_prompts):
            prompt_to_embedding[p] = batch_embeds[i].tolist()

        elapsed = time.time() - start_time
        done = end
        avg_time = elapsed / max(done, 1)
        remaining = avg_time * (total_unique - done)
        print(f"Embedded {done}/{total_unique} unique prompts | Elapsed: {elapsed:.1f}s | ETA: {remaining/60:.1f}m")

    # --- EXPAND BACK TO ORIGINAL ITEMS ---
    results = []
    for item in data:
        p = item["prompt"]
        results.append({
            "index": item["index"],
            "prompt_embed": prompt_to_embedding[p],
        })

    print(f"Total time taken: {time.time() - start_time:.1f} seconds")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Wrote embeddings to: {output_path}")


if __name__ == "__main__":
    main()
