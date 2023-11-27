import torch
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM

import optimum
from optimum.intel.openvino import OVModelForCausalLM
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Callable, Iterable
import numpy as np
import torch
from optimum.intel.openvino import OVModelForCausalLM
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from openvino.runtime import Model, Core, Tensor, Type
from optimum.intel.openvino.utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME
from transformers import PretrainedConfig
from tempfile import TemporaryDirectory
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig, StoppingCriteriaList, AutoConfig,AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers.generation.utils import GenerateOutput

def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.
    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    # if tokens do not match continue
                    match = True
                    break
            stopped_samples.append(match)

        return stopped_samples


class OVQwenModel(OVModelForCausalLM):
    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf['qwen'] = NormalizedTextConfig.with_args(
            num_layers='num_layers', num_attention_heads='num_attention_heads')
        self.is_first_forward = True
        super().__init__(model, config, device, dynamic_shapes,
                         ov_config, model_save_dir, **kwargs)

    def _reshape(
        self,
        model: Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        init_cls = OVQwenModel

        return init_cls(model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get("past", None)

        # `past_key_values` may be in the stardard format (e.g. in contrastive search), converts to bloom's format if needed
        if past_key_values is not None and self.config.model_type == "bloom":
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()
        inputs = {}

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][1].shape[-2]
            if self._pkv_precision == Type.bf16:
                # numpy does not support bf16, pretending f16, should change to bf16
                past_key_values = tuple(
                    Tensor(past_key_value, past_key_value.shape, Type.bf16)
                    for pkv_per_layer in past_key_values
                    for past_key_value in pkv_per_layer
                )
            else:
                # Flatten the past_key_values
                past_key_values = tuple(
                    past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )

            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            batch_size = input_ids.shape[0]
            if self.config.model_type == "bloom":
                batch_size *= self.normalized_config.num_attention_heads

            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = batch_size
                if shape[2].is_dynamic:
                    shape[2] = 0
                else:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs["input_ids"] = np.array(input_ids)
        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = np.expand_dims(position_ids[:, -1], axis=-1)

            inputs["position_ids"] = position_ids

        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )

        if self.is_first_forward:
            #print("logits.shape: ", logits.shape)
            self.logits_list = logits
            self.is_first_forward = False
            #print("logits.shape: ", logits.shape)
        """
        else:
            #print("logits.shape: ", logits.shape)
            #print("self.logits_list.shape before cat: ", self.logits_list.shape)
            self.logits_list = torch.cat((self.logits_list, logits), dim=1)
            #print("self.logits_list.shape after cat: ", self.logits_list.shape)
        """

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _update_model_kwargs_for_generation(
        self,
        outputs: "ModelOutput",
        model_kwargs: Dict[str, "Any"],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, "Any"]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat([position_ids, new_position_id], dim=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs


    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        #assistant_model: Optional["PreTrainedModel"] = None,
        #streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        # Process stop_words_ids.
        stop_words_ids = kwargs.pop("stop_words_ids", [[151643]])
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            **kwargs,
        )

class OVChatGLM2Model(OVModelForCausalLM):
    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf['chatglm'] = NormalizedTextConfig.with_args(
            num_layers='num_layers', num_attention_heads='num_attention_heads')
        self.is_first_forward = True
        super().__init__(model, config, device, dynamic_shapes,
                         ov_config, model_save_dir, **kwargs)
    def _reshape(
        self,
        model: Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith('past_key_values'):
                shapes[inputs][1] = -1
                shapes[inputs][2] = 2
            else:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        #print("forward called!!!")
        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            if self._pkv_precision == Type.bf16:
                # numpy does not support bf16, pretending f16, should change to bf16
                past_key_values = tuple(
                    Tensor(past_key_value, past_key_value.shape, Type.bf16) for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )
            else:
                # Flatten the past_key_values
                past_key_values = tuple(
                    past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = 0
                if shape[1].is_dynamic:
                    shape[1] = 1
                inputs[input_name] = Tensor(
                    model_inputs.get_element_type(), shape.get_shape())

        inputs['input_ids'] = np.array(input_ids)

        if 'position_ids' in self.input_names and position_ids is not None:
            inputs['position_ids'] = np.array(position_ids)

        # Add the attention_mask inputs when needed
        if 'attention_mask' in self.input_names and attention_mask is not None:
            inputs['attention_mask'] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

        logits = torch.from_numpy(
            self.request.get_tensor('logits').data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(
                key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i: i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
        else:
            past_key_values = None

        if self.is_first_forward:
            #print("logits.shape: ", logits.shape)
            self.logits_list = logits
            self.is_first_forward = False
            #print("logits.shape: ", logits.shape)
        """
        else:
            #print("logits.shape: ", logits.shape)
            #print("self.logits_list.shape before cat: ", self.logits_list.shape)
            self.logits_list = torch.cat((self.logits_list, logits), dim=1)
            #print("self.logits_list.shape after cat: ", self.logits_list.shape)
        """
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        #print("prepare_inputs_for_generation call!!! ")
        past_key_values = past_key_values or kwargs.get("past", None)

        # `past_key_values` may be in the stardard format (e.g. in contrastive search), converts to bloom's format if needed
        if past_key_values is not None and self.config.model_type == "bloom":
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs: "ModelOutput",
        model_kwargs: Dict[str, "Any"],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, "Any"]:
        #print("_update_model_kwargs_for_generation!!!")
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        init_cls = OVChatGLM2Model

        return init_cls(model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs)

class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            # Initialize new model and tokenizer instances
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(self.device)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        else:
            raise TypeError(
                "Parameter pretrained should be of type str or transformers.PreTrainedModel"
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM

class OPTIMUMLM(BaseLM):
    def __init__(
        self,
        device="cpu",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__()

        #import optimum
        #from optimum.intel.openvino import OVModelForCausalLM

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int,str))

        device_list = set(["cuda", "cpu"] + [f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")
        """
        self.gpt2 = OVModelForCausalLM.from_pretrained(
            pretrained,
            load_in_8bit=load_in_8bit,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_cache=True,
        )
        """

        self.config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=True)

        print("pretrained: ", pretrained)
        if self.config.model_type == "llama":
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    pretrained if tokenizer is None else tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
            )
            except:
                print("Tokenizer is missed. Please save it into the same folder with the model.")
            self.gpt2 = OVModelForCausalLM.from_pretrained(
                pretrained,
                load_in_8bit=load_in_8bit,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_cache=True,
            )


        elif self.config.model_type == "chatglm":
            self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained,
                    padding_side='left',
                    trust_remote_code=True)

            self.gpt2 = OVChatGLM2Model.from_pretrained(
                pretrained,
                config=self.config,
                #load_in_8bit=load_in_8bit,
                #revision=revision,
                pad_token_id=self.tokenizer.pad_token_id,
                trust_remote_code=trust_remote_code,
            )

        elif self.config.model_type == "qwen":
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained,
                pad_token='<|extra_0|>',
                eos_token='<|endoftext|>',
                padding_side='left',
                trust_remote_code=True)

            self.gpt2 = OVQwenModel.from_pretrained(pretrained,
                                                    config=self.config,
                                                    compile=False,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    trust_remote_code=True)
            self.gpt2.generation_config = GenerationConfig.from_pretrained(pretrained, pad_token_id=self.tokenizer.pad_token_id)

        self.gpt2.logits_list = []
        """
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except:
            print("Tokenizer is missed. Plaase save it into the same folder with the model.")
        """

        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == 'auto': 
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size) 

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            if self.config.model_type == "chatglm" or self.config.model_type == "qwen":
                return self.gpt2.config.seq_length # Hack for ChatGLM2/3,Qwen
            else:
                # gptneoconfig doesn't have n_ctx apparently
                return self.gpt2.config.max_position_embeddings


    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        #print("_model_call !!!")
        ret = None
        if self.config.model_type == "chatglm" or self.config.model_type == "qwen":
            generation_kwargs = {'do_sample': False, 'max_new_tokens': 1}
            if self.tokenizer.eos_token_id is not None:
                generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
                generation_kwargs['pad_token_id'] = self.tokenizer.eos_token_id # setting eos_token_id as pad token
            self.gpt2.is_first_forward = True
            self.gpt2.logits_list = []
            self.gpt2.generate(inps, **generation_kwargs)
            ret = self.gpt2.logits_list
        else:
            #return self.gpt2(inps)[0]
            ret = self.gpt2(inps)[0]

        return ret

    def _model_generate(self, context, max_length, eos_token_id):
        #print("_model_generate !!!")
        generation_kwargs = {'do_sample': False, 'max_length': max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.gpt2.generate(context, **generation_kwargs)
