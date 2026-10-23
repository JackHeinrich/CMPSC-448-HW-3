import torch

import copy

from data import build_vocab, encode

from model import TransformerModel

import time

TRAIN_PERCENTAGE = 0.9

ROUNDING_PRECISION = 3

SAVE_INTERVAL = 1000

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Return a random batch of inputs x and next-character targets y, each (batch_size, block_size).
def get_batch(data, block_size, batch_size):

    starts = torch.randint(len(data) - block_size - 1, (batch_size,))

    x_list = []
    y_list = []

    for s in starts:
        x_list.append(data[s : s + block_size])
        y_list.append(data[s + 1 : s + block_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x.to(device), y.to(device)

# Get current loss on some data
@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, iters=50):
    was_training = model.training # remember if model was training
    model.eval() # turn off dropout for a fair measurement
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training) # put the model back the way we found it
    return sum(losses) / len(losses) # Return average loss

def save_model(path, state, encoder, decoder, config):
    torch.save({
        "model": state,
        "char_to_int": encoder,
        "int_to_char": decoder,
        "config": config,
    }, path)

def train(dataset: str, config: dict, model_name: str = 'model'):
    encoder, decoder = build_vocab(dataset)

    encoded_dataset = encode(encoder, dataset)

    data = torch.tensor(encoded_dataset)

    train_split = int(TRAIN_PERCENTAGE * len(data))

    train_data = data[:train_split]

    val_data = data[train_split:]

    block_size = config["block_size"]
    batch_size = config["batch_size"]

    model = TransformerModel(len(encoder), block_size, config["vector_size"], config["num_heads"], config["num_blocks"], config["dropout"]).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    start_time = time.perf_counter()

    best_val = float("inf")

    best_state = None

    for step in range(config["training_steps"]):

        x, y = get_batch(train_data, block_size, batch_size)

        _, loss = model(x, y)

        optimizer.zero_grad() # clear gradients so they dont accumulate

        loss.backward() # backprop to find new gradients

        optimizer.step() # move each weight against gradient

        if step % 100 == 0:
            train_loss = estimate_loss(model, train_data, block_size, batch_size)
            val_loss = estimate_loss(model, val_data, block_size, batch_size)
            print(step, "train loss: ", round(train_loss, ROUNDING_PRECISION), "val loss: ", round(val_loss, ROUNDING_PRECISION))

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

        if step > 0 and step % SAVE_INTERVAL == 0:
            save_model(f"{model_name}_checkpoint.pt", best_state, encoder, decoder, config)
            print(f"Checkpoint saved to {model_name}_checkpoint.pt")

    elapsed = time.perf_counter() - start_time

    print(f"Trained model {model_name} for {config['training_steps']} steps. Total elapsed time: {elapsed:.1f}s")

    final_train = estimate_loss(model, train_data, block_size, batch_size)
    final_val = estimate_loss(model, val_data, block_size, batch_size)
    print("final", "train", round(final_train, ROUNDING_PRECISION), "val", round(final_val, ROUNDING_PRECISION))

    if final_val < best_val:
        best_val = final_val
        best_state = copy.deepcopy(model.state_dict())

    print(f"Best val loss: {best_val:.3f}")

    while True:
        answer = input("Want to save? (y/n): ").strip().lower()
        if answer == "y":
            save_model(f"{model_name}.pt", best_state, encoder, decoder, config)
            print(f"Saved to {model_name}.pt")
            break
        elif answer == "n":
            print("Not saved.")
            break

if __name__ == "__main__":

    print(f"Training on: {device}.")

    data_path = input("Enter dataset path (.txt): ")

    model_name = input("Enter model name: ")

    training_steps = int(input("Enter training steps (integer): "))

    block_size = int(input("Enter block size / context length (e.g. 128): "))

    batch_size = int(input("Enter batch size (e.g. 32): "))

    vector_size = int(input("Enter vector size (e.g. 128): "))

    num_heads = int(input("Enter number of attention heads (e.g. 4): "))

    if vector_size % num_heads != 0:
            print("Vector size must be divisible by number of heads.")
            exit()

    num_blocks = int(input("Enter number of transformer blocks (e.g. 4): "))

    dropout = float(input("Enter dropout (e.g. 0.1): "))

    learning_rate = float(input("Enter learning rate (e.g. 0.001): "))

    config = {
        "training_steps": training_steps,
        "block_size": block_size,
        "batch_size": batch_size,
        "vector_size": vector_size,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "dropout": dropout,
        "learning_rate": learning_rate,
    }

    with open(data_path, encoding="utf-8") as f:
        dataset = f.read()

    train(dataset, config, model_name)