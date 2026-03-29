import logging
import textwrap
from dataclasses import dataclass
from typing import ClassVar

import torch
from transformers import AutoConfig, ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

from .dataset import EveryQueryBatch

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
    logger.info("FlashAttention is available.")
except ImportError:
    HAS_FLASH_ATTN = False


BRANCH = "│ "


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


class MLP(torch.nn.Module):
    """Multi-layer perceptron with ReLU activations and optional dropout.

    Examples:
        >>> mlp = MLP([64, 32, 1], dropout_prob=0.1)
        >>> x = torch.randn(4, 64)
        >>> mlp(x).shape
        torch.Size([4, 1])

        Two-layer (no hidden layer, no dropout):

        >>> mlp2 = MLP([8, 1], dropout_prob=0.5)
        >>> mlp2(torch.randn(2, 8)).shape
        torch.Size([2, 1])
    """

    def __init__(self, layers, dropout_prob):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            modules.append(torch.nn.ReLU())
            if i < len(layers) - 2:
                modules.append(torch.nn.Dropout(dropout_prob))
        modules.pop(-1)  # we don't want an activation after the final layer
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


@dataclass
class EveryQueryOutput(BaseModelOutput):
    """Extended output for EveryQuery that includes task-specific fields.

    Inherits base fields like `last_hidden_state`, `hidden_states`, and `attentions` from
    Hugging Face's `BaseModelOutput`, and adds:
      - `query_embed`: Pooled embedding used for downstream heads.
      - `censor_logits`: Logits for the censor prediction head.
      - `censor_loss`: Loss for the censor prediction.
      - `occurs_logits`: Logits for the occurs prediction head.
      - `occurs_loss`: Loss for the occurs prediction.
    """

    query_embed: torch.FloatTensor | None = None
    censor_logits: torch.FloatTensor | None = None
    censor_loss: torch.FloatTensor | None = None
    occurs_logits: torch.FloatTensor | None = None
    occurs_loss: torch.FloatTensor | None = None

    def __shape_str_lines(self) -> list[str]:
        """Generates the lines for the shape block."""

        shape_lines: list[str] = ["Shape:"]

        if self.last_hidden_state is not None:
            try:
                batch_size, seq_len, hidden_size = self.last_hidden_state.shape
                shape_lines.append(f"{BRANCH}Batch size: {batch_size}")
                shape_lines.append(f"{BRANCH}Sequence length: {seq_len}")
                shape_lines.append(f"{BRANCH}Hidden size: {hidden_size}")
            except Exception:
                shape_lines.append(f"{BRANCH}last_hidden_state: {tuple(self.last_hidden_state.shape)}")

        if self.query_embed is not None:
            shape_lines.append(f"{BRANCH}Query embedding: {tuple(self.query_embed.shape)}")

        if self.hidden_states is not None:
            try:
                num_layers = len(self.hidden_states)
                example_shape = tuple(self.hidden_states[0].shape)
                shape_lines.append(f"{BRANCH}Hidden states: {num_layers} layers by {example_shape}")
            except Exception:
                shape_lines.append(f"{BRANCH}Hidden states: {type(self.hidden_states)}")

        if self.attentions is not None:
            try:
                num_layers = len(self.attentions)
                example_shape = tuple(self.attentions[0].shape)
                shape_lines.append(f"{BRANCH}Attentions: {num_layers} layers by {example_shape}")
            except Exception:
                shape_lines.append(f"{BRANCH}Attentions: {type(self.attentions)}")

        return shape_lines

    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to probabilities via sigmoid, squeezing trailing dimensions.

        Examples:
            >>> EveryQueryOutput.logits_to_probs(torch.tensor([[0.0]]))
            tensor(0.5000)
            >>> EveryQueryOutput.logits_to_probs(torch.zeros(3, 1))
            tensor([0.5000, 0.5000, 0.5000])

            Saturates near 0 and 1 for extreme logits:

            >>> EveryQueryOutput.logits_to_probs(torch.tensor([[1000.0]])) > 0.999
            tensor(True)
            >>> EveryQueryOutput.logits_to_probs(torch.tensor([[-1000.0]])) < 0.001
            tensor(True)
        """
        return torch.sigmoid(logits).squeeze()

    @property
    def occurs_probs(self) -> torch.Tensor | None:
        """Sigmoid probabilities for the occurs head, or ``None`` if logits are absent.

        Examples:
            >>> EveryQueryOutput(last_hidden_state=None).occurs_probs is None
            True
            >>> out = EveryQueryOutput(last_hidden_state=None, occurs_logits=torch.zeros(3, 1))
            >>> out.occurs_probs
            tensor([0.5000, 0.5000, 0.5000])
        """
        if self.occurs_logits is None:
            return None
        return self.logits_to_probs(self.occurs_logits)

    @property
    def censor_probs(self) -> torch.Tensor | None:
        """Sigmoid probabilities for the censor head, or ``None`` if logits are absent.

        Examples:
            >>> EveryQueryOutput(last_hidden_state=None).censor_probs is None
            True
            >>> out = EveryQueryOutput(last_hidden_state=None, censor_logits=torch.zeros(2, 1))
            >>> out.censor_probs
            tensor([0.5000, 0.5000])
        """
        if self.censor_logits is None:
            return None
        return self.logits_to_probs(self.censor_logits)

    @staticmethod
    def __str_tensor_val(tensor: torch.Tensor) -> str:
        """Strips the `tensor(` prefix, `)` suffix, leading/trailing , and newlines."""

        tensor_str = str(tensor).replace("tensor(", "       ").replace(")", "")
        tensor_str = "\n".join([x for x in tensor_str.splitlines() if x.strip()])
        tensor_str = textwrap.dedent(tensor_str).strip()
        return tensor_str

    def __str_tensor_list(self, header: str, tensors: list[str]) -> list[str]:
        """Gets string representation lines for the requested tensors (by attribute name)."""
        out: list[str] = [f"{header}:"]
        for tensor_n in tensors:
            tensor = getattr(self, tensor_n, None)
            if tensor is None:
                continue

            dtype = getattr(tensor, "dtype", None)
            dtype_str = f" ({dtype})" if dtype is not None else ""

            out.append(f"{BRANCH}{tensor_n}{dtype_str}:")
            try:
                tensor_str = self.__str_tensor_val(tensor)
            except Exception:
                tensor_str = str(tensor)
            out.extend(textwrap.indent(tensor_str, BRANCH + BRANCH).splitlines())

        return out

    def __data_str_lines(self) -> list[str]:
        """Generates the lines for the data block."""
        data_lines: list[str] = ["Data:"]

        core = self.__str_tensor_list(
            "Core",
            [
                "last_hidden_state",
                "query_embed",
            ],
        )
        if len(core) > 1:
            data_lines.extend([f"{BRANCH}{line}" for line in core])

        heads = self.__str_tensor_list(
            "Heads",
            [
                "censor_logits",
                "occurs_logits",
            ],
        )
        if len(heads) > 1:
            data_lines.append(BRANCH)
            data_lines.extend([f"{BRANCH}{line}" for line in heads])

        losses = self.__str_tensor_list(
            "Losses",
            [
                "censor_loss",
                "occurs_loss",
            ],
        )
        if len(losses) > 1:
            data_lines.append(BRANCH)
            data_lines.extend([f"{BRANCH}{line}" for line in losses])

        return data_lines

    def __str__(self) -> str:
        """Human-readable string representation (for debugging and doctests)."""

        lines: list[str] = [f"{self.__class__.__name__}:"]

        torch.set_printoptions(precision=2, threshold=5, edgeitems=2)

        lines.extend([f"{BRANCH}{line}" for line in self.__shape_str_lines()])
        lines.append(BRANCH)
        lines.extend([f"{BRANCH}{line}" for line in self.__data_str_lines()])

        torch.set_printoptions(profile="default")

        lines = [line.rstrip() for line in lines]
        return "\n".join(lines)


class EveryQueryModel(torch.nn.Module):
    HF_model_config: ModernBertConfig
    HF_model: ModernBertModel
    do_demo: bool
    do_grad_ckpt: bool
    precision: str
    mlp_dropout: float

    PRECISION_TO_MODEL_WEIGHTS_DTYPE: ClassVar[dict[str, torch.dtype]] = {
        "32-true": torch.float32,
        "16-true": torch.float16,
        "16-mixed": torch.float32,
        "bf16-true": torch.bfloat16,
        "bf16-mixed": torch.float32,
        "transformer-engine": torch.bfloat16,
    }

    def __init__(
        self,
        precision: str = "32-true",
        do_demo: bool = False,
        do_grad_ckpt: bool = False,
        mlp_dropout: float = 0.1,
        model_name_or_config: str | ModernBertConfig = "answerdotai/ModernBERT-base",
    ):
        super().__init__()

        if isinstance(model_name_or_config, ModernBertConfig):
            self.HF_model_config = model_name_or_config
        else:
            self.HF_model_config: ModernBertConfig = AutoConfig.from_pretrained(model_name_or_config)

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

        self.HF_model_config.output_hidden_states = False
        self.HF_model_config.output_attentions = False
        self.HF_model_config.use_cache = False
        self.HF_model_config.mlp_dropout = float(mlp_dropout)

        self.HF_model = ModernBertModel._from_config(self.HF_model_config, **extra_kwargs)

        self.do_grad_ckpt = do_grad_ckpt
        if self.do_grad_ckpt and hasattr(self.HF_model, "gradient_checkpointing_enable"):
            self.HF_model.gradient_checkpointing_enable()
            # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one

        self.censor_mlp = MLP(
            layers=[self.HF_model.config.hidden_size, 128, 1], dropout_prob=self.HF_model.config.mlp_dropout
        )
        self.occurs_mlp = MLP(
            layers=[self.HF_model.config.hidden_size, 128, 1], dropout_prob=self.HF_model.config.mlp_dropout
        )
        self.duration_embed = MLP(layers=[1, 64, self.HF_model.config.hidden_size], dropout_prob=0)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.do_demo = do_demo
        if self.do_demo:
            self.forward = self._forward_demo
        else:
            self.forward = self._forward

        self.hparams = {
            "precision": precision,
            "do_demo": do_demo,
            "mlp_dropout": self.HF_model.config.mlp_dropout,
            "model_name_or_config": model_name_or_config,
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
            >>> valid = Mock(mode="SM", code=torch.tensor([[1, 2, 3]]), duration_days=None, PAD_INDEX=0)
            >>> demo_model._check_inputs(valid)

            >>> bad_mode = Mock(mode="JNRT")
            >>> demo_model._check_inputs(bad_mode)
            Traceback (most recent call last):
                ...
            ValueError: Batch mode JNRT is not supported.

            >>> short = Mock(mode="SM", code=torch.tensor([[5]]), duration_days=None, PAD_INDEX=0)
            >>> demo_model._check_inputs(short)
            Traceback (most recent call last):
                ...
            ValueError: Input sequence length 1 is too short. Minimum sequence length is 2.

            >>> oov = Mock(mode="SM", code=torch.tensor([[1, 200]]), duration_days=None, PAD_INDEX=0)
            >>> demo_model._check_inputs(oov)
            Traceback (most recent call last):
                ...
            AssertionError: ...out-of-vocabulary...

            Sequence exceeding ``max_position_embeddings`` (128 for the demo model):

            >>> long_code = torch.ones(1, 129, dtype=torch.long)
            >>> demo_model._check_inputs(Mock(mode="SM", code=long_code, duration_days=None, PAD_INDEX=0))
            Traceback (most recent call last):
                ...
            ValueError: ...exceeds model max sequence length...

            Duration token adds +1, so seq_len=128 trips the limit:

            >>> demo_model._check_inputs(Mock(mode="SM", code=torch.ones(1, 128, dtype=torch.long), duration_days=torch.tensor([30.0]), PAD_INDEX=0))
            Traceback (most recent call last):
                ...
            ValueError: ...exceeds model max sequence length...

            All-padding batch:

            >>> demo_model._check_inputs(Mock(mode="SM", code=torch.zeros(1, 3, dtype=torch.long), duration_days=None, PAD_INDEX=0))
            Traceback (most recent call last):
                ...
            AssertionError: ...only padding tokens...
        """

        code = batch.code

        if batch.mode != "SM":
            raise ValueError(f"Batch mode {batch.mode} is not supported.")

        _batch_size, seq_len = code.shape
        effective_seq_len = seq_len + (1 if batch.duration_days is not None else 0)

        if effective_seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {effective_seq_len} (including duration token) exceeds model max "
                f"sequence length {self.max_seq_len}."
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
            Without ``duration_days``, returns ``input_ids`` and ``attention_mask``:

            >>> no_dur = Mock(mode="SM", code=torch.tensor([[1, 2, 3], [4, 5, 0]]), duration_days=None, PAD_INDEX=0)
            >>> hf_in = demo_model._hf_inputs(no_dur)
            >>> sorted(hf_in.keys())
            ['attention_mask', 'input_ids']
            >>> hf_in['attention_mask']
            tensor([[ True,  True,  True],
                    [ True,  True, False]])

            With ``duration_days``, embeds a duration token at position 1:

            >>> with torch.no_grad():
            ...     hf_dur = demo_model._hf_inputs(sample_batch)
            >>> sorted(hf_dur.keys())
            ['attention_mask', 'inputs_embeds']
            >>> hf_dur['inputs_embeds'].shape[1] == sample_batch.code.shape[1] + 1
            True
            >>> hf_dur['inputs_embeds'].shape[2] == demo_model_config.hidden_size
            True
            >>> hf_dur['attention_mask'].shape[1] == sample_batch.code.shape[1] + 1
            True
        """
        attention_mask = batch.code != batch.PAD_INDEX  # (batch_size, seq_len)

        if batch.duration_days is not None:
            # tok_embeddings in newer transformers, word_embeddings in older
            embed_layer = getattr(self.HF_model.embeddings, "tok_embeddings", None) or getattr(
                self.HF_model.embeddings, "word_embeddings"
            )
            word_embeds = embed_layer(batch.code)  # (B, seq_len, H)
            dur_norm = (batch.duration_days / 365.0).unsqueeze(-1)  # (B, 1)
            dur_emb = self.duration_embed(dur_norm).unsqueeze(1)  # (B, 1, H)
            # Insert duration embedding at position 1 (after query token at position 0)
            inputs_embeds = torch.cat([word_embeds[:, :1, :], dur_emb, word_embeds[:, 1:, :]], dim=1)
            dur_mask = torch.ones(batch.code.shape[0], 1, dtype=torch.bool, device=batch.code.device)
            attention_mask = torch.cat([attention_mask[:, :1], dur_mask, attention_mask[:, 1:]], dim=1)
            return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

        return {
            "input_ids": batch.code,
            "attention_mask": attention_mask,
        }

    def _forward_demo(self, batch: EveryQueryBatch) -> tuple[torch.FloatTensor, BaseModelOutput]:
        """A demo forward pass that adds more checks and assertions."""

        self._check_inputs(batch)
        self._check_parameters()
        out = self._forward(batch)
        self._check_outputs(*out)

        return out

    def _get_loss(self, logits, target, mask=None):
        """Computes BCE-with-logits loss, optionally masked.

        Examples:
            >>> logits = torch.tensor([[0.5], [-0.5]])
            >>> target = torch.tensor([1, 0])
            >>> loss = demo_model._get_loss(logits, target)
            >>> loss.shape
            torch.Size([])
            >>> loss.isfinite()
            tensor(True)

            >>> mask = torch.tensor([True, False])
            >>> demo_model._get_loss(logits, target, mask=mask).isfinite()
            tensor(True)

            Known value: logit 0, target 1 gives -log(sigmoid(0)) = log(2):

            >>> round(demo_model._get_loss(torch.tensor([[0.0]]), torch.tensor([1])).item(), 4)
            0.6931
        """
        target = target.float().unsqueeze(1)
        if mask is not None:
            logits = logits[mask]
            target = target[mask]
        assert logits.shape == target.shape, f"logits: {logits.shape}, target: {target.shape}"
        return self.criterion(logits, target)

    def _forward(self, batch: EveryQueryBatch) -> tuple[torch.FloatTensor, BaseModelOutput]:
        """Runs the model forward pass: HF backbone, query pooling, task heads, and loss.

        Examples:
            >>> with torch.no_grad():
            ...     loss, outputs = demo_model._forward(sample_batch)
            >>> loss.isfinite()
            tensor(True)
            >>> isinstance(outputs, EveryQueryOutput)
            True
            >>> outputs.query_embed.shape == (sample_batch.batch_size, demo_model_config.hidden_size)
            True
            >>> outputs.censor_logits.shape == (sample_batch.batch_size, 1)
            True
            >>> outputs.occurs_logits.shape == (sample_batch.batch_size, 1)
            True

            Loss decomposes into the two sub-losses:

            >>> outputs.censor_loss.isfinite() and outputs.occurs_loss.isfinite()
            tensor(True)
            >>> torch.allclose(loss, outputs.censor_loss + outputs.occurs_loss)
            True

            ``do_demo=True`` preserves hidden states in the output:

            >>> outputs.last_hidden_state is not None
            True
        """
        outputs = self.HF_model(**self._hf_inputs(batch))
        embeddings = outputs.last_hidden_state  # (batch_size, seq_len + query_len, hidden_size)
        query_embed = embeddings[:, 0, :]  # 0 is query_index (batch_size, hidden_size)

        censor_logits = self.censor_mlp(query_embed)
        censor_loss = self._get_loss(censor_logits, batch.censor, mask=None)

        # if future is censored then don't apply loss on whether query occurs
        occurs_logits = self.occurs_mlp(query_embed)
        occurs_loss = self._get_loss(occurs_logits, batch.occurs, mask=~batch.censor)

        loss = censor_loss + occurs_loss

        outputs = EveryQueryOutput(
            last_hidden_state=outputs.last_hidden_state if self.do_demo else None,
            hidden_states=outputs.hidden_states if self.do_demo else None,
            attentions=outputs.attentions if self.do_demo else None,
            query_embed=query_embed,
            censor_logits=censor_logits,
            censor_loss=censor_loss,
            occurs_logits=occurs_logits,
            occurs_loss=occurs_loss,
        )

        return loss, outputs
