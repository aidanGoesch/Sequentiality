# from src.sequentiality import calculate_sequentiality  # LLM (loprob) Sequentiality
from src.embedding import calculate_sequentiality  # USE Sequentiality
import os
import sys
import pandas as pd

# Models:
"microsoft/Phi-3-mini-4k-instruct"
"SakanaAI/TinySwallow-1.5B-Instruct"
"meta-llama/Llama-3.3-70B-Instruct"
"meta-llama/Llama-3.2-3B-Instruct"

# non-prompt finetuned
"openai-community/gpt2-xl"
"allenai/OLMo-2-1124-13B"

# models that are used in the ensemble
MODEL_IDS = ["SakanaAI/TinySwallow-1.5B-Instruct",
            "openai-community/gpt2-xl",
            "allenai/OLMo-2-1124-13B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct"]


def example(model_id: int | None):
    """
    Example invocation of Sequentiality metric. 

    model_id can specify an index in MODEL_IDS or can be None if using USE
    """
    dataset = pd.read_csv("/path/to/csv")

    # list of history lengths
    history_lengths = list(range(1, 10))

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=history_lengths,
                                     text_input=list(dataset["story"]),
                                     topics=list(dataset["topic"]))
    

    # Save to outputs folder - change savepath here
    os.makedirs("./output/", exist_ok=True)
    
    # clean model id
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./output/{safe_model_name}.csv")


def example1(model_id: int):
    """
    Example invocation of Sequentiality metric
    """
    dataset = pd.read_csv("./hcV3-tri-topic.csv")

    # list of history lengths
    history_lengths = list(range(1, 10))

    output = calculate_sequentiality(model=model_id,
                                     history_lengths=history_lengths,
                                     text_input=list(dataset["story"]),
                                     topics=list(dataset["mainEvent"]))
    

    # Save to outputs folder - change savepath here
    os.makedirs("./output/", exist_ok=True)
    
    # clean model id
    safe_model_name = model_id.replace("/", "_")
    output.to_csv(f"./output/{safe_model_name}.csv")

# Example usage:
if __name__ == "__main__":
    model_idx = int(sys.argv[1])
    if model_idx in range(len(MODEL_IDS)):
        model = MODEL_IDS[model_idx]
    else:
        print("invalid model index")
        exit(-2)

    example1(model)
