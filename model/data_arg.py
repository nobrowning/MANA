class DataArgs:
    def __init__(self, src_data_size, src_tgt_key, src_num_feats,
                 dst_data_size, dst_tgt_key, dst_num_feats):
        self.src_data_size = src_data_size
        self.src_tgt_key = src_tgt_key
        self.src_num_feats = src_num_feats
        
        self.dst_data_size = dst_data_size
        self.dst_tgt_key = dst_tgt_key
        self.dst_num_feats = dst_num_feats