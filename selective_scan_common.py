import torch


def SSMScanOp(ab0, ab1):
    return [ab0[0] * ab1[0], ab1[0] * ab0[1] + ab1[1]]


class SSMScanPrefixCallbackOp:
    def __init__(self, init_val: list):
        self.running_prefix = init_val

    def __call__(self, x: int):
        old_prefix = self.running_prefix.copy()
        self.running_prefix = SSMScanOp(self.running_prefix, x)
        return old_prefix


# Note in the orginal implementation there is another parameter which gives the thread access
# to shared memory to help it store the value faster, but we'll just ignore that for now
def load_input(
    u: torch.Tensor, threadData: torch.Tensor, temp_storage: torch.Tensor, seqLen: int
): ...


# Note in the orginal implementation there is another parameter which gives the thread access
# to shared memory to help it store the value faster, but we'll just ignore that for now
def load_weight(
    u: torch.Tensor, threadData: torch.Tensor, temp_storage: torch.Tensor, seqLen: int
): ...


# Note in the orginal implementation there is another parameter which gives the thread access
# to shared memory to help it store the value faster, but we'll just ignore that for now
def store_input(
    u: torch.Tensor, threadData: torch.Tensor, temp_storage: torch.Tensor, seqLen: int
): ...
