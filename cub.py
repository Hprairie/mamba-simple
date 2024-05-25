class GPUPrefixScan:
    def __init__(self, shared_mem):
        self.shared_mem = shared_mem

    def InclusiveScan(self, input_data, output_data, scan_op, prefix_op):
        """
        Check out cub::BlockScan to understand what this object is
        """
        pass
