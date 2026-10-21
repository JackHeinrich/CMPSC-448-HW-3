import torch

from data import build_vocab, encode, decode

DATASET_PATH = './pg1661.txt'

TRAIN_PERCENTAGE = 0.9

BLOCK_SIZE = 128

BATCH_SIZE = 32

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

    return x, y


if __name__ == "__main__":

    dataset = None

    with open(DATASET_PATH, encoding='UTF-8') as f:
        dataset = f.read()

    encoder, decoder = build_vocab(dataset)

    encoded_dataset = encode(encoder, dataset)

    data = torch.tensor(encoded_dataset)

    train_split = int(TRAIN_PERCENTAGE * len(data))

    train_data = data[:train_split]

    val_data = data[train_split:]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    x, y = get_batch(train_data)
    