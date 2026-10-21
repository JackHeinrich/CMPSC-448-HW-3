# Build lookup tables for character to integer and integer back to character for encoding / decoding
def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:

    chars = sorted(set(text))

    encoder = {}
    decoder = {}

    for i in range(len(chars)):
        char = chars[i]
        encoder[char] = i
        decoder[i] = char

    return encoder, decoder

# Encode a string into a list of integers
def encode(encoder: dict[str, int], text: str) -> list[int]:
    encoded_text = []

    for char in text:
        encoded_text.append(encoder[char])

    return encoded_text

# Decode list of integers back into a string
def decode(decoder: dict[int, str], encoded_text: list[int]) -> str:

    decoded_text = ''.join(decoder[num] for num in encoded_text)

    return decoded_text

# Test everything
if __name__ == "__main__":
    encoder, decoder = build_vocab("Holmes")
    print(encode(encoder, "Holmes"))
    print(decode(decoder, encode(encoder, "Holmes")))
