import torch

from data import build_vocab, encode, decode

from model import TransformerModel

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

    for step in range(training_steps):

        x, y = get_batch(train_data)

        _, loss = model(x, y)

        optimizer.zero_grad() # clear gradients so they dont accumulate

        loss.backward() # backprop ti find new gradients

        optimizer.step() # move each weight with gradient

        if step % 100 == 0:
            print(step, "train", round(estimate_loss(model, train_data), ROUNDING_PRECISION), "val", round(estimate_loss(model, val_data), ROUNDING_PRECISION))

    print("final", "train", round(estimate_loss(model, train_data), ROUNDING_PRECISION), "val", round(estimate_loss(model, val_data), ROUNDING_PRECISION))

    should_save = input("Want to save? (y/n): ")
    if should_save.strip().lower() == "y":
        torch.save({
            "model": model.state_dict(),
            "char_to_int": encoder,
            "int_to_char": decoder,
            "block_size": BLOCK_SIZE,
        }, f"{model_name}.pt")
        print("Saved to sherlock.pt")

if __name__ == "__main__":
    with open("./pg1661.txt", encoding="utf-8") as f:
        dataset = f.read()
    train(dataset, 10, "sherlock")