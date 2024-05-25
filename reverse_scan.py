class GPUPostfixScan:
    def __init__(self, shared_mem):
        self.shared_mem = shared_mem

    def InclusiveScan(self, input_data, output_data, scan_op, prefix_op):
        """
        Check out cub::BlockScan to understand what this object does.

        However do note that cub::BlockScan is for a prefix scan, while Tri Dao and Albert Gu write their own postfix scan.

            -> This just does the scan in reverse order
            -> Saves us from having to reverse the input across threads
        """
        pass
