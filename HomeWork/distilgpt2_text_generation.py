import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "distilgpt2"  # A small, fast GPT-2 model


# 1. Setup Model and Tokenizer
def setup_lm():
    """Initializes the tokenizer and the pre-trained language model."""
    print(f"Setting up Language Model: {MODEL_NAME}...")

    # The tokenizer handles converting raw text into the numerical IDs the model understands.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # The model is loaded with pre-trained weights for text generation (Causal LM).
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()  # Set to evaluation mode

    print("Model and Tokenizer loaded successfully.")
    return model, tokenizer


# 2. Prepare Input
def prepare_text_input(tokenizer, prompt_text):
    """Encodes the input text string into a PyTorch tensor of token IDs."""
    print(f"\nEncoding Prompt: '{prompt_text}'")

    # The tokenizer encodes the text. return_tensors='pt' ensures PyTorch tensor output.
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    return input_ids


# 3. Run Inference
def run_generation(model, tokenizer, input_ids, max_length=150):
    """Generates new text based on the input tokens."""
    print("Running text generation...")

    # Generates text. max_length specifies how long the final output sequence should be.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.6,
            top_k=50,
            top_p=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return output_ids


# 4. Display Results
def display_results(tokenizer, generated_output_ids, prompt_length):
    """Decodes the output tensor into human-readable text."""

    # Decode the output tensor back into a string.
    full_text = tokenizer.decode(generated_output_ids[0], skip_special_tokens=True)

    # Separate the original prompt from the generated text
    original_prompt = full_text[:prompt_length]
    generated_text = full_text[prompt_length:].strip()

    print("\n--- Generation Results ---")
    print(f"Original Prompt: {original_prompt}")
    print("-" * 30, "\n")
    print(f"Generated Text: {generated_text}")
    print("-" * 30, "\n")


def main():
    model, tokenizer = setup_lm()

    # Text Input
    PROMPT = "The future of AI in education is"

    input_ids = prepare_text_input(tokenizer, PROMPT)

    # Get the length of the prompt in tokens for cleaner display
    prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))

    output_ids = run_generation(model, tokenizer, input_ids, max_length=50)

    display_results(tokenizer, output_ids, prompt_length)


if __name__ == "__main__":
    main()