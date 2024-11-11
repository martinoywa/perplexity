import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset


logging.set_verbosity_debug()    # logging to DEBUG


class Perplexity:
    """
        Class for calculating the perplexity or average of Negative Log-likelihood (NLL) of a sequence for
        a model.
    """
    def __init__(self, model_id, stride, device):
        self.stride = int(stride)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encodings = self.load_encodings()

    
    def load_encodings(self, return_type="pt"):
        text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return self.tokenizer("\n\n".join(text["text"]), return_tensors=return_type)
    

    def calculate_perplexity(self):
        max_length = self.model.config.n_positions
        seq_len = self.encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, self.stride)):
            end_loc = min(begin_loc+max_length, seq_len)
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(self.device)

            target_ids = input_ids.clone()
            target_len = end_loc - prev_end_loc
            target_ids[:, :-target_len] = -100  # first tokens of size stride will be -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over target_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl
