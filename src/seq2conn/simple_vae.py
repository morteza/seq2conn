from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Encoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        out, (h, _) = self.encoder(x)
        h = h.permute(1, 0, 2)
        return h


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Decoder, self).__init__()

        self.decoder = nn.LSTM(hidden_dim, input_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, h):
        out, (h, _) = self.decoder(h)
        h = h.permute(1, 0, 2)
        out = self.fc(out)
        return out, h


class SimpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_subjects=1):
        super(SimpleVAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, n_layers)
        self.decoder = Decoder(input_dim, hidden_dim, n_layers)

    def forward(self, x):
        h_enc = self.encoder(x)
        out, h_dec = self.decoder(None, h_enc)
        return out
