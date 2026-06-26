import torch


class UTF8Encoder:

    @staticmethod
    def encode(text: str):

        # utf8 bytes
        byte_data = text.encode("utf-8")

        # convert to integer list
        ids = list(byte_data)

        return torch.tensor(ids, dtype=torch.long)


# example

# tensor([72,101,108,108,111,...])