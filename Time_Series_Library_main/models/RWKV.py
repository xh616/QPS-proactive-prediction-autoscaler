import torch


class TimeMixing(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.u = torch.nn.Parameter(torch.ones(1, 1, dim))
        self.w = torch.nn.Parameter(torch.ones(1, 1, dim))

        self.time_mix_receptance = torch.nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_key = torch.nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_value = torch.nn.Parameter(torch.ones(1, 1, dim))

        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))  # shifts data one step to the right
        self.sigmoid = torch.nn.Sigmoid()

        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.receptance = torch.nn.Linear(dim, dim, bias=False)

        self.ln_out = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, T, d = x.shape
        x_shifted = self.time_shift(x)

        key = x * self.time_mix_key + x_shifted * (1 - self.time_mix_key)
        value = x * self.time_mix_value + x_shifted * (1 - self.time_mix_value)
        receptance = self.sigmoid(x * self.time_mix_receptance + x_shifted * (1 - self.time_mix_receptance))

        key, value, receptance = self.key(key), self.value(value), self.receptance(receptance)

        wkv = torch.zeros_like(key)  # (B T d)
        a_t = torch.zeros_like(key[:, 0, :])  # (B d)
        b_t = torch.zeros_like(key[:, 0, :])  # (B d)

        for i in range(T):
            q = torch.maximum(self.u + key[:, i, :], self.w)

            a_t = torch.exp(-self.w - q) * a_t + torch.exp(self.u + key[:, i, :] - q) * value[:, i, :]
            b_t = torch.exp(-self.w - q) * b_t + torch.exp(self.u + key[:, i, :] - q)

            wkv[:, i, :] = a_t / b_t

        return self.ln_out(wkv * receptance)


class ChannelMixing(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.channel_mix_receptance = torch.nn.Parameter(torch.ones(1, 1, dim))
        self.channel_mix_key = torch.nn.Parameter(torch.ones(1, 1, dim))

        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))  # shifts data one step to the right
        self.sigmoid = torch.nn.Sigmoid()

        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        self.receptance = torch.nn.Linear(dim, dim, bias=False)

        self.ln_out = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, T, d = x.shape
        x_shifted = self.time_shift(x)

        key = x * self.channel_mix_key + x_shifted * (1 - self.channel_mix_key)
        receptance = x * self.channel_mix_receptance + x_shifted * (1 - self.channel_mix_receptance)

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = self.sigmoid(self.receptance(receptance))

        return receptance * value


class Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()


        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)

        self.time_mixing = TimeMixing(dim)
        self.channel_mixing = ChannelMixing(dim)

        self.ln_out = torch.nn.LayerNorm(dim)

    def forward(self, x):
        # print(x.shape)
        attention = self.time_mixing(self.ln1(x))
        x = x + attention

        ffn = self.channel_mixing(self.ln2(x))
        x = x + ffn

        return x

class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.enc_in

        self.rwkv = torch.nn.ModuleList([
            Block(dim) for _ in range(config.e_layers)
        ])
        # self.ln_out = torch.nn.LayerNorm(dim)
        # self.ln_in = torch.nn.LayerNorm(dim)
        # self.embed = torch.nn.Linear(config.enc_in, config.d_model)
        self.linear1 = torch.nn.Linear(config.seq_len, config.d_ff)
        self.activate = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(config.d_ff, config.pred_len)

        # self.linear1 = torch.nn.Linear(config.d_model, config.d_ff)
        # self.activate = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        # self.linear2 = torch.nn.Linear(config.d_ff, config.enc_in)

    def forward(self, x, batch_x_mark=None, dec_inp=None, batch_y_mark=None):
        # x = self.ln_in(x)

        # x = self.embed(x)

        for rwkv_block in self.rwkv:
            x = rwkv_block(x)

        # x = self.ln_out(x)
        x = self.dropout(x)

        x = self.dropout(self.linear1(x.transpose(1, -1)))
        x = self.activate(x)
        x = self.dropout(self.linear2(x).transpose(1, -1))
        return x