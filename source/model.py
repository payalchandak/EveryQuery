import logging
from typing import ClassVar

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput
from dataset import EveryQueryBatch

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
    logger.info("FlashAttention is available.")
except ImportError:
    HAS_FLASH_ATTN = False


def _val(tensor: torch.Tensor) -> int | bool | float:
    """Returns the value of a scalar-tensor as a Python scalar.

    Examples:
        >>> _val(torch.tensor(1))
        1
        >>> _val(torch.tensor(1.0))
        1.0
        >>> _val(torch.tensor(False))
        False
    """
    return tensor.detach().cpu().item()


class Model(torch.nn.Module):

    HF_model_config: ModernBertConfig
    HF_model: ModernBertModel
    do_demo: bool
    precision: str

    PRECISION_TO_MODEL_WEIGHTS_DTYPE: ClassVar[dict[str, torch.dtype]] = {
        "32-true": torch.float32,
        "16-true": torch.float16,
        "16-mixed": torch.float32,
        "bf16-true": torch.bfloat16,
        "bf16-mixed": torch.float32,
        "transformer-engine": torch.bfloat16,
    }

    def __init__(self, precision: str = "32-true", do_demo: bool = False):
        super().__init__()

        self.HF_model_config: ModernBertConfig = AutoConfig.from_pretrained("answerdotai/ModernBERT-large")

        extra_kwargs = {"torch_dtype": self.PRECISION_TO_MODEL_WEIGHTS_DTYPE.get(precision)}

        if HAS_FLASH_ATTN:
            logger.info("Using FlashAttention 2 for the model.")
            extra_kwargs["attn_implementation"] = "flash_attention_2"

            if precision in {"16-mixed", "bf16-mixed"}:
                logger.info(
                    "Using mixed precision for Flash Attention 2.0. Ignore the warning that may appear "
                    "below about Flash Attention 2.0 only supporting torch.float16 and torch.bfloat16. "
                    "Lightning will automatically cast the model to the correct dtype during training in "
                    "mixed precision mode."
                )
            elif precision not in {"16-true", "bf16-true", "transformer-engine"}:
                logger.warning(
                    "Flash Attention 2.0 is only supported for precision '16-true', 'bf16-true', "
                    f"'transformer-engine', '16-mixed' and 'bf16-mixed'. Using {precision} may cause errors."
                )

        self.HF_model = ModernBertModel._from_config(self.HF_model_config, **extra_kwargs)

        self.do_demo = do_demo
        if self.do_demo:
            self.forward = self._forward_demo
        else:
            self.forward = self._forward

        self.hparams = {
            "precision": precision,
            "do_demo": do_demo,
        }

    @property
    def max_seq_len(self) -> int:
        """The maximum sequence length of the model."""
        return self.HF_model_config.max_position_embeddings

    @property
    def vocab_size(self) -> int:
        """The vocabulary size of the model."""
        return self.HF_model_config.vocab_size

    def _check_inputs(self, batch: EveryQueryBatch):
        """Checks the inputs for various validity properties.

        Validity checks:
          - The batch is in "SM" mode (see
            [MEDSTorchBatch](https://meds-torch-data.readthedocs.io/en/latest/api/meds_torchdata/types/#meds_torchdata.types.MEDSTorchBatch)
            for more details).
          - The input sequence length must not exceed the model's maximum sequence length.
          - The input sequence length must not be too short (minimum sequence length is 2).
          - The input must not contain out-of-vocabulary tokens.
          - The input must not contain inf or nan values.
          - The input must not contain only padding tokens.

        Args:
            batch: The input batch of data.

        Raises:
            ValueError: If the input sequence length exceeds the model's maximum sequence length.
            AssertionError: If the input contains out-of-vocabulary tokens or if it contains inf or nan
                values.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> batch = Mock(code=torch.LongTensor([[0, 3, 1], [0, 2, 1]]), PAD_INDEX=0, mode="SM")
            >>> model._check_inputs(batch) # no errors
            >>> batch.code = torch.LongTensor([[0, 3, 1], [0, 2, 11]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: Input sequence contains 1 out-of-vocabulary tokens (max 11 for vocab size 10).
            >>> batch.code = torch.Tensor([[0, 3, 1], [0, 2, float("inf")]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: Batch code contains inf values.
            >>> batch.code = torch.Tensor([[0, 3, 1], [0, 2, float("nan")]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: Batch code contains nan values.
            >>> batch.code = torch.LongTensor([[0, 3, 1], [0, 0, 0]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            AssertionError: 1 samples in the batch have only padding tokens. Batch size: 2, Sequence length: 3
            >>> batch.code = torch.LongTensor([[0, 3, 1, 0], [0, 2, 1, 4]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            ValueError: Input sequence length 4 exceeds model max sequence length 3.
            >>> batch.code = torch.LongTensor([[1], [2]])
            >>> model._check_inputs(batch)
            Traceback (most recent call last):
                ...
            ValueError: Input sequence length 1 is too short. Minimum sequence length is 2.
            >>> model._check_inputs(Mock(mode="SEM"))
            Traceback (most recent call last):
                ...
            ValueError: Batch mode SEM is not supported.
        """

        code = batch.code

        if batch.mode != "SM":
            raise ValueError(f"Batch mode {batch.mode} is not supported.")

        batch_size, seq_len = code.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {batch.code.shape[1]} exceeds model max sequence length "
                f"{self.max_seq_len}."
            )
        elif seq_len <= 1:
            raise ValueError(
                f"Input sequence length {batch.code.shape[1]} is too short. Minimum sequence length is 2."
            )

        torch._assert(~torch.isinf(code).any(), "Batch code contains inf values.")
        torch._assert(~torch.isnan(code).any(), "Batch code contains nan values.")

        out_of_vocab = code >= self.vocab_size
        out_of_vocab_msg = (
            f"Input sequence contains {out_of_vocab.sum()} out-of-vocabulary tokens "
            f"(max {batch.code.max()} for vocab size {self.vocab_size})."
        )

        torch._assert(~out_of_vocab.any(), out_of_vocab_msg)

        is_pad = code == batch.PAD_INDEX  # Shape is batch_size x seq_len
        all_samples_pad = is_pad.all(dim=1)  # Shape is batch_size

        all_samples_pad_msg = (
            f"{all_samples_pad.sum()} samples in the batch have only padding tokens. "
            f"Batch size: {code.shape[0]}, Sequence length: {code.shape[1]}"
        )
        torch._assert(~all_samples_pad.any(), all_samples_pad_msg)

    def _check_parameters(self):
        """Logs a warning about the finiteness of any parameters in the model.

        This is only used for advanced debugging. It does not raise an error because when this mode is
        enabled, typically detect anomaly is on in the lightning trainer, and that gives more information
        about these issues than a generic assertion would.

        Validity checks:
            - The parameters are not nan.
            - The parameters are not inf.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> model.HF_model.gpt_neox.layers[1].attention.query_key_value.bias.shape
            torch.Size([12])
            >>> model.HF_model.gpt_neox.layers[1].attention.query_key_value.bias = torch.nn.Parameter(
            ...     torch.tensor([float("nan"), 0., float("inf"), 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            ... )
            >>> with print_warnings():
            ...     model._check_parameters()
            Warning: Parameter HF_model.gpt_neox.layers.1.attention.query_key_value.bias contains 1/12 nan
                values.
            Warning: Parameter HF_model.gpt_neox.layers.1.attention.query_key_value.bias contains 1/12 inf
                values.
        """

        for n, p in self.named_parameters():
            num_nan = _val(torch.isnan(p).sum())
            num_inf = _val(torch.isinf(p).sum())

            if num_nan > 0:
                logger.warning(f"Parameter {n} contains {num_nan}/{p.numel()} nan values.")
            if num_inf > 0:
                logger.warning(f"Parameter {n} contains {num_inf}/{p.numel()} inf values.")

    def _check_outputs(self, loss: torch.FloatTensor, outputs: BaseModelOutput):
        """Logs a warning if the loss is inf or nan.

        This does not raise an error because when this mode is enabled, typically detect anomaly is on in the
        lightning trainer, and that gives more information about these issues than a generic assertion would.

        Validity checks:
            - The loss is not inf or nan.
            - The logits contain inf or nan values.

        Args:
            loss: The loss tensor.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> fake_output_valid = Mock(logits=torch.FloatTensor([[0.1, 0.2], [0.3, 0.4]]))
            >>> model._check_outputs(torch.tensor(0.4), fake_output_valid) # no errors
            >>> with print_warnings():
            ...     model._check_outputs(torch.tensor(float("inf")), fake_output_valid)
            ...     model._check_outputs(torch.tensor(float("nan")), fake_output_valid)
            Warning: Loss contains inf values.
            Warning: Loss contains nan values.
            >>> fake_output_inf = Mock(logits=torch.FloatTensor([[float("inf"), 0.2], [0.3, 0.4]]))
            >>> fake_output_nan = Mock(logits=torch.FloatTensor([[0.4, float("nan")], [0.3, float("nan")]]))
            >>> with print_warnings():
            ...     model._check_outputs(torch.tensor(0.4), fake_output_inf)
            ...     model._check_outputs(torch.tensor(0.4), fake_output_nan)
            Warning: Logits contains 1/4 inf values.
            Warning: Logits contains 2/4 nan values.
        """

        if _val(torch.isinf(loss).any()):
            logger.warning("Loss contains inf values.")
        if _val(torch.isnan(loss).any()):
            logger.warning("Loss contains nan values.")

        embeddings = outputs.last_hidden_state
        inf_count = _val(torch.isinf(embeddings).sum())
        if inf_count > 0:
            logger.warning(f"Embeddings contains {inf_count}/{embeddings.numel()} inf values.")
        nan_count = _val(torch.isnan(embeddings).sum())
        if nan_count > 0:
            logger.warning(f"Embeddings contains {nan_count}/{embeddings.numel()} nan values.")

    def _hf_inputs(self, batch: EveryQueryBatch) -> dict[str, torch.Tensor]:
        """Converts the EveryQueryBatch to a dictionary of inputs for the Hugging Face model.

        HF relevant input keys:
            - input_ids: The input sequence of token IDs. Captured in `batch.code`.
            - attention_mask: A mask to avoid attending to padding tokens. See the
              [documentation](https://huggingface.co/docs/transformers/en/model_doc/gpt_neox#transformers.GPTNeoXModel.forward.attention_mask)
              for more details. Should be a tensor of shape `(batch_size, seq_len)` (same as `input_ids`) with
              0s for tokens that are masked and 1s for tokens that are not masked. This means it is given by
              `batch.code != batch.PAD_INDEX` as whenever the code is not a padding token, it should be
              attended to.

        Args:
            batch: The input batch of data.

        Returns:
            A dictionary of inputs for the Hugging Face model.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> batch = Mock(code=torch.LongTensor([[0, 3, 1], [0, 0, 2]]), PAD_INDEX=0)
            >>> model._hf_inputs(batch)
            {'input_ids': tensor([[0, 3, 1],
                                  [0, 0, 2]]),
             'attention_mask': tensor([[False,  True,  True],
                                       [False, False,  True]])}
        """
        return {
            "input_ids": batch.code,
            "attention_mask": (batch.code != batch.PAD_INDEX),
        }

    def _forward_demo(self, batch: EveryQueryBatch) -> tuple[torch.FloatTensor, BaseModelOutput]:
        """A demo forward pass that adds more checks and assertions."""

        self._check_inputs(batch)
        self._check_parameters()
        out = self._forward(batch)
        self._check_outputs(*out)

        return out

    def _forward(self, batch: EveryQueryBatch) -> tuple[torch.FloatTensor, BaseModelOutput]:
        outputs = self.HF_model(**self._hf_inputs(batch))
        embeddings = outputs.last_hidden_state
        loss = None

        return loss, outputs

