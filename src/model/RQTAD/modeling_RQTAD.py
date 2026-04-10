from transformers import PreTrainedModel
from torch import nn
import torch



class Resbook(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_books = config.num_books
        self.book_size = config.book_size
        self.feature_dim = config.feature_dim
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(self.book_size[i], self.feature_dim[i])for i in range(self.num_books)
            ]
        )
        self.peride = config.peride

        for i in range(self.num_books):
            nn.init.zeros_(self.codebooks[i].weight)
            self.codebooks[i].weight.requires_grad = False

    def rotate_by_phase(self, X, idx):
        r = idx % self.peride
        return torch.roll(X, shifts=-r, dims=1)
    
    def 

    def fit(self, timeseries, timestamp):

    def predict(self, timeseries, timestamp):

        





class RQTAD(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.codebook = Resbook(config)





    def forward(self, timeseries, timestamp, **kwargs):
        return