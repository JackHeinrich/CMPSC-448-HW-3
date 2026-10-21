import torch

import copy

from data import build_vocab, encode, decode

from model import TransformerModel

import time

TRAIN_PERCENTAGE = 0.9

BLOCK_SIZE = 128

BATCH_SIZE = 32

ROUNDING_PRECISION = 3

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Return a random batch of inputs x and next-character targets y, each (BATCH_SIZE, BLOCK_SIZE).
def get_batch(data):

    starts = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))

    x_list = []
    y_list = []

    for s in starts:
        x_list.append(data[s : s + BLOCK_SIZE])
        y_list.append(data[s + 1 : s + BLOCK_SIZE + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x.to(device), y.to(device)

# Get current loss on some data
@torch.no_grad()
def estimate_loss(model, data, iters=50):
    was_training = model.training # remember if model was training
    model.eval() # turn off dropout for a fair measurement
    losses = []
    for _ in range(iters):
        x, y = get_batch(data)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training) # put the model back the way we found it
    return sum(losses) / len(losses) # Return average loss

def train(dataset: str, training_steps: int, model_name: str = 'model'):
    encoder, decoder = build_vocab(dataset)

    encoded_dataset = encode(encoder, dataset)

    data = torch.tensor(encoded_dataset)

    train_split = int(TRAIN_PERCENTAGE * len(data))

    train_data = data[:train_split]

    val_data = data[train_split:]

    model = TransformerModel(len(encoder), BLOCK_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Small learning rate to keep learning stable

    start_time = time.perf_counter()

    best_val = float("inf")

    best_state = None

    for step in range(training_steps):

        x, y = get_batch(train_data)

        _, loss = model(x, y)

        optimizer.zero_grad() # clear gradients so they dont accumulate

        loss.backward() # backprop to find new gradients

        optimizer.step() # move each weight against gradient

        if step % 100 == 0:
            train_loss = estimate_loss(model, train_data)
            val_loss = estimate_loss(model, val_data)
            print(step, "train loss: ", round(train_loss, ROUNDING_PRECISION), "val loss: ", round(val_loss, ROUNDING_PRECISION))

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

    elapsed = time.perf_counter() - start_time

    print(f"Trained model {model_name} for {training_steps} steps. Total elapsed time: {elapsed:.1f}s")

    final_train = estimate_loss(model, train_data)
    final_val = estimate_loss(model, val_data)
    print("final", "train", round(final_train, ROUNDING_PRECISION), "val", round(final_val, ROUNDING_PRECISION))

    if final_val < best_val:
        best_val = final_val
        best_state = copy.deepcopy(model.state_dict())

    print(f"Best val loss: {best_val:.3f}")

    while True:
        answer = input("Want to save? (y/n): ").strip().lower()
        if answer == "y":
            torch.save({
                "model": best_state,
                "char_to_int": encoder,
                "int_to_char": decoder,
                "block_size": BLOCK_SIZE,
            }, f"{model_name}.pt")
            print(f"Saved to {model_name}.pt")
            break
        elif answer == "n":
            print("Not saved.")
            break

if __name__ == "__main__":

    print(f"Training on: {device}.")

    data_path = input("Enter dataset path (.txt): ")

    steps_input = input("Enter training steps (integer): ")

    training_steps = int(steps_input)

    model_name = input("Enter model name: ")

    with open(data_path, encoding="utf-8") as f:
        dataset = f.read()

    train(dataset, training_steps, model_name)