from perpelexity import Perplexity
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Calcualte LLM Perplexity using HuggingFace.")
    parser.add_argument("--hf_model_id", type=str, help="HuggingFace Model ID")
    parser.add_argument("--stride", type=str, help="Sliding Window Stride. Recommended MODEL_CONTEXT_WINDOW//2.")
    parser.add_argument("--device", type=str, help="Device model is running i.e gpu/cpu/mps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pinecone_indexer = Perplexity(args.hf_model_id, args.stride, args.device)
    pinecone_indexer.calculate_perplexity()
