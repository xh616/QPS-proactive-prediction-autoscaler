class Model_Param:
    def __init__(self, seq_len=120, pred_len=60, e_layers=2, d_layers=1, factor=3, enc_in=4, dec_in=4, c_out=4, d_ff=256, dropout=0.1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.factor = factor
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_ff = d_ff
        self.dropout = dropout