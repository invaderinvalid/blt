import torch


class PatchBuilder:

    def __init__(self, threshold=3.5):

        self.threshold = threshold

    def build(
        self,
        embeddings,
        entropy
    ):

        """
        embeddings = [seq, dim]
        entropy = [seq]
        
        Returns:
            patches (Tensor): Shape [num_patches, dim]
            boundaries (list): End index of each patch (0-indexed)
        """

        patches = []
        boundaries = []
        current = []

        for i in range(
            len(entropy)
        ):

            current.append(
                embeddings[i]
            )

            # split point
            if entropy[i] > self.threshold:

                patch = torch.stack(
                    current
                ).mean(dim=0)

                patches.append(
                    patch
                )
                
                boundaries.append(i)

                current = []

        if len(current) > 0:

            patch = torch.stack(
                current
            ).mean(dim=0)

            patches.append(
                patch
            )
            
            boundaries.append(len(entropy) - 1)

        return torch.stack(patches), boundaries