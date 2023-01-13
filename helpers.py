from pathlib import Path
from typing import Dict

from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel


class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True


class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """

    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.beasjdklaj.parameters():
            p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    # def forward(self,
    #             query: Dict[str, torch.LongTensor],
    #             document: Dict[str, torch.LongTensor]):

    #     query_vecs = self.forward_representation(query)
    #     document_vecs = self.forward_representation(document)

    #     score = self.forward_aggregation(query_vecs,document_vecs,query["attention_mask"],document["attention_mask"])
    #     return score

    def forward(self, input_ids, attention_mask, sequence_type=None) -> torch.Tensor:

        vecs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]  # assuming a distilbert model here
        vecs = self.compressor(vecs)

        # if encoding only, zero-out the mask values so we can compress storage
        # if sequence_type == "doc_encode" or sequence_type == "query_encode":
        #     vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask):

        # create initial term-x-term scores (dot-product)
        score = torch.bmm(query_vecs, document_vecs.transpose(2, 1))

        # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0 might select them)
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = -10000

        # max pooling over document dimension
        score = score.max(-1).values

        # mask out paddding query values
        score[~(query_mask.bool())] = 0

        # sum over query values
        score = score.sum(-1)

        return score


class AmazonCat13K(torch.utils.data.Dataset):
    def __init__(self, root: str = "~/data/AmazonCat-13K", train: bool = False, tokenizer: str | None = None) -> None:
        self.root = Path(root).expanduser()
        self.file_name = "trn.json.gz" if train else "tst.json.gz"

        # load dataframe
        self.df = pd.read_json(self.root / self.file_name, compression="gzip", lines=True).set_index("uid")

        # load tokenizer (optional)
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            self.tokenizer = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor] | str:
        text = self.df["content"][index]
        if self.tokenizer is not None:
            return self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
        else:
            return text


def flattened_to_batched(
    a: np.ndarray, batch_indices: np.ndarray, padding: int | None = None, return_att_mask: bool = False
) -> np.ndarray:
    """Restore the batched view of subsequent sequences.
    The returned view will be padded to the longest occuring sequence.

    `N` is the cumulative number of sequence elements over all sequences

    Args:
        a (np.ndarray): Array containing sequences subsequently - shape (N, ...)
        batch_indices (np.ndarray): Array containing batch indices for each sequence element
    """
    # ensure batch_indices are sequential from 0 to max
    _, batch_indices = np.unique(batch_indices, return_inverse=True)
    # indices of the values in a
    seq_indices = np.arange(len(batch_indices))
    # compute mask where the start of every sequence is 'True'
    seq_starts = np.concatenate([[True], (batch_indices[:-1] != batch_indices[1:])])
    # compute the sequence offsets 'x[seq_starts][index]' and subtract them from the sequence_indices
    # (this resets the indexing at sequence beginning)
    seq_indices = seq_indices - seq_indices[seq_starts][batch_indices]

    # batch size
    n = batch_indices.max() + 1
    # maximum sequence length
    if isinstance(padding, int):
        seq_len = padding
    else:
        seq_len = seq_indices.max() + 1

    # the output array for the batched view
    out = np.zeros((n, seq_len, *a.shape[1:]))
    # write the values from flattened view to the batched view array
    out[batch_indices, seq_indices] = a

    # return batched view
    if return_att_mask:
        attention_mask = np.zeros((n, seq_len))
        attention_mask[batch_indices, seq_indices] = 1

        return out, attention_mask
    else:
        return out
