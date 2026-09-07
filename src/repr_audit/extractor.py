"""
Hidden state extraction from HuggingFace transformer models.

Handles both encoder models (BERT, RoBERTa) and decoder models
(GPT-2, Llama, Mistral) with appropriate pooling strategies and token-level extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ModelFamily = Literal["encoder", "decoder"]


@dataclass
class ExtractionConfig:
    """Configuration for hidden state extraction.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier (e.g. "bert-base-uncased").
    batch_size:
        Number of sentences per forward pass.
    device:
        Torch device string. Defaults to "cuda" if available, else "cpu".
    pooling:
        Token pooling strategy for sentence-level representations.
        - "mean"       : average all non-padding tokens (standard for encoders)
        - "cls"        : use [CLS] token (BERT-style)
        - "last_token" : use last non-padding token (standard for decoders)
    max_length:
        Maximum tokenization length.
    padding_side:
        "left" or "right". If None, automatically configured based on model family
        ("left" for decoders to prevent causal mask corruption, "right" for encoders).
    """

    model_name: str
    batch_size: int = 32
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    pooling: Literal["mean", "cls", "last_token"] = "mean"
    max_length: int = 128
    padding_side: Literal["left", "right"] | None = None


class HiddenStateExtractor:
    """Extracts layer-wise hidden states and token embeddings from a transformer model.

    Parameters
    ----------
    config:
        Extraction configuration or model name string.

    Examples
    --------
    >>> extractor = HiddenStateExtractor("bert-base-uncased")
    >>> states = extractor.extract(["The cat sat.", "A dog ran."])
    >>> states.shape  # (num_layers+1, num_sentences, hidden_size)
    """

    def __init__(self, config: ExtractionConfig | str) -> None:
        if isinstance(config, str):
            config = ExtractionConfig(model_name=config)
        self.config = config
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model onto the configured device with correct padding."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=False)

        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            output_hidden_states=True,
            torch_dtype=torch.float32,
        ).to(self.config.device)
        self.model.eval()

        # Detect family
        family = self.detect_model_family()

        # Add pad token for decoder-only models that lack one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set padding side: decoders require left-padding for causal attention and last_token pooling
        if self.config.padding_side is not None:
            self.tokenizer.padding_side = self.config.padding_side
        elif family == "decoder":
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.padding_side = "right"

    @property
    def num_layers(self) -> int:
        """Number of transformer layers (excluding embedding layer)."""
        cfg = self.model.config
        val = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
        if val is None:
            # Fallback for unconventional configs
            val = getattr(cfg, "num_layers", 12)
        return int(val)

    def extract(self, sentences: list[str]) -> np.ndarray:
        """Extract pooled hidden states for every layer.

        Parameters
        ----------
        sentences:
            List of raw text strings.

        Returns
        -------
        np.ndarray of shape (num_layers + 1, num_sentences, hidden_size).
        Layer 0 is the embedding layer output; layers 1..N are transformer layers.
        """
        all_layer_reps: list[list[np.ndarray]] = []

        for batch_start in tqdm(
            range(0, len(sentences), self.config.batch_size),
            desc=f"Extracting [{self.config.model_name}]",
            leave=False,
        ):
            batch = sentences[batch_start : batch_start + self.config.batch_size]
            batch_reps = self._extract_batch(batch)  # (num_layers+1, batch, hidden)

            if not all_layer_reps:
                all_layer_reps = [[] for _ in range(batch_reps.shape[0])]
            for layer_idx in range(batch_reps.shape[0]):
                all_layer_reps[layer_idx].append(batch_reps[layer_idx])

        stacked = np.stack(
            [np.concatenate(layer_batches, axis=0) for layer_batches in all_layer_reps],
            axis=0,
        )
        return stacked

    def _extract_batch(self, batch: list[str]) -> np.ndarray:
        """Run a single batch through the model and pool representations."""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden)
        attention_mask = inputs["attention_mask"]

        layer_reps = []
        for layer_hidden in hidden_states:
            pooled = self._pool(layer_hidden, attention_mask)
            layer_reps.append(pooled.cpu().float().numpy())

        return np.stack(layer_reps, axis=0)  # (num_layers+1, batch, hidden)

    def _pool(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling strategy to get sentence representations."""
        strategy = self.config.pooling

        if strategy == "cls":
            # In right-padded encoder models, CLS is token 0
            # If left-padded, find first non-padding token or use position 0 if right-padded
            if self.tokenizer.padding_side == "left":
                # First non-padding token
                first_tokens = (attention_mask == 1).int().argmax(dim=1)
                batch_size = hidden.size(0)
                return hidden[torch.arange(batch_size, device=hidden.device), first_tokens]
            return hidden[:, 0, :]

        elif strategy == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

        elif strategy == "last_token":
            # For left-padded decoders, last non-padding token is at the final sequence position (-1)
            if self.tokenizer.padding_side == "left":
                return hidden[:, -1, :]
            # For right-padded decoders, compute last non-padding position explicitly
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden.size(0)
            return hidden[torch.arange(batch_size, device=hidden.device), seq_lengths]

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy!r}")

    def extract_token_representations(
        self,
        sentences: list[str],
        target_word: str,
    ) -> np.ndarray:
        """Extract layer-wise contextualized representations of a specific target word.

        Locates the character span of `target_word` in each sentence, maps it to the
        constituent subword token indices using tokenizer offset mappings, and mean-pools
        the subwords into a single representation for the word occurrence.

        Parameters
        ----------
        sentences:
            List of sentences containing the target word.
        target_word:
            Target token / lemma (e.g. "bank").

        Returns
        -------
        np.ndarray of shape (num_layers + 1, num_found_occurrences, hidden_size).
        """
        all_layer_reps: list[list[np.ndarray]] = []
        pattern = re.compile(rf"\b{re.escape(target_word)}\b", re.IGNORECASE)

        for batch_start in range(0, len(sentences), self.config.batch_size):
            batch = sentences[batch_start : batch_start + self.config.batch_size]

            # Tokenize with offsets
            try:
                encodings = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_offsets_mapping=True,
                )
                offset_mappings = encodings.pop("offset_mapping").cpu().numpy()
            except Exception:
                # Fallback if fast tokenizer offsets unavailable
                encodings = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                )
                offset_mappings = None

            inputs = {k: v.to(self.config.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states  # (num_layers+1) tuple of (batch, seq, hidden)

            num_layers = len(hidden_states)
            if not all_layer_reps:
                all_layer_reps = [[] for _ in range(num_layers)]

            # For each sentence in batch, locate the target word tokens
            for b_idx, text in enumerate(batch):
                match = pattern.search(text)
                if match is None:
                    # Fallback to substring search
                    start_char = text.lower().find(target_word.lower())
                    if start_char == -1:
                        continue
                    end_char = start_char + len(target_word)
                else:
                    start_char, end_char = match.start(), match.end()

                token_indices = []
                if offset_mappings is not None:
                    offsets = offset_mappings[b_idx]
                    for t_idx, (t_start, t_end) in enumerate(offsets):
                        if t_start == 0 and t_end == 0:
                            continue  # Special token
                        if t_start < end_char and t_end > start_char:
                            token_indices.append(t_idx)

                # Fallback: if offsets mapping not present or produced empty indices
                if not token_indices:
                    # Subword matching fallback
                    input_ids = inputs["input_ids"][b_idx].cpu().tolist()
                    target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
                    # Search for target_ids sub-sequence
                    for i in range(len(input_ids) - len(target_ids) + 1):
                        if input_ids[i : i + len(target_ids)] == target_ids:
                            token_indices = list(range(i, i + len(target_ids)))
                            break

                if not token_indices:
                    # Target word was not tokenized cleanly; skip occurrence
                    continue

                # Pool subwords at each layer
                for l in range(num_layers):
                    subword_vecs = hidden_states[l][
                        b_idx, token_indices, :
                    ]  # (num_subwords, hidden)
                    token_vec = subword_vecs.mean(dim=0).cpu().float().numpy()  # (hidden,)
                    all_layer_reps[l].append(token_vec)

        if not all_layer_reps or len(all_layer_reps[0]) == 0:
            # If word was never found, return empty array
            hidden_dim = getattr(self.model.config, "hidden_size", 768)
            num_layers_total = self.num_layers + 1
            return np.empty((num_layers_total, 0, hidden_dim), dtype=np.float32)

        stacked = np.stack(
            [np.array(layer_list, dtype=np.float32) for layer_list in all_layer_reps],
            axis=0,
        )
        return stacked

    def detect_model_family(self) -> ModelFamily:
        """Detect whether the model is encoder or decoder architecture."""
        cfg = self.model.config
        model_type = getattr(cfg, "model_type", "").lower()
        decoder_types = {
            "gpt2",
            "gpt_neo",
            "gpt_neox",
            "llama",
            "mistral",
            "falcon",
            "gemma",
            "qwen2",
            "bloom",
            "opt",
        }
        if model_type in decoder_types or getattr(cfg, "is_decoder", False):
            return "decoder"
        return "encoder"
