import torch

from data import encode, decode
from model import TransformerModel

MAX_NEW_TOKENS = 300

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the saved weights + vocab and rebuild the model
def load_model(path):
    checkpoint = torch.load(path, map_location=device) # map_location: load on whatever device this machine has
    encoder, decoder = checkpoint["char_to_int"], checkpoint["int_to_char"]
    model = TransformerModel(len(encoder), checkpoint["block_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval() # dropout off for generation
    return model, encoder, decoder

# clean prompts to prevent crashes -- the book uses curly quotes and unknown chars would crash encode()
def clean_prompt(prompt, encoder):
    prompt = prompt.replace('"', "“").replace("'", "’")
    return "".join(ch for ch in prompt if ch in encoder)

if __name__ == "__main__":
    path = input(f"Enter model path: ")

    model, encoder, decoder = load_model(path)

    print(f"Loaded {path}. Type a prompt (or 'quit').")

    while True:

        prompt = input("\nEnter prompt: ")

        if prompt.strip().lower() == "quit":
            break

        prompt = clean_prompt(prompt, encoder)

        if not prompt:
            prompt = "\n" # empty prompt -- start from a newline

        encoded = torch.tensor([encode(encoder, prompt)], dtype=torch.long, device=device) # shape (1, len)
        output = model.generate(encoded, MAX_NEW_TOKENS)
        print(decode(decoder, output[0].tolist()))