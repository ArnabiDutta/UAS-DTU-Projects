# Audio Classification Model
## Flow
Transcript Generation With Whisper --> Tokenization --> Word Embeddings [Training] --> Classification [Training]
### 1. Whisper Model
* Creating a Pipeline
  ```
  model_path= "/home/cha0s/Desktop/arnabi/whisper/temp_dir_1"
  processor = WhisperProcessor.from_pretrained(model_path)
  model = WhisperForConditionalGeneration.from_pretrained(model_path,use_safetensors=True)
  model.to(device)
  # model.generation_config = generation_config
  print("Lesgoo model loaded!")
  audio_path="/home/cha0s/Desktop/arnabi/whisper/audio/What__I_can’t_hear_anything!_There’s_smoke,_and_I’m_trapped_in_the_debris!.wav"
  feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
  
  asr_pipeline = pipeline(
      "automatic-speech-recognition",
      model=model,
      tokenizer=processor.tokenizer,
      feature_extractor=feature_extractor,
      device=0,
      chunk_length_s=30,
      return_timestamps=False
  )

#### Other Pipeline Params
```
def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    processor: Optional[Union[str, ProcessorMixin]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map=None,
    torch_dtype=None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
)
```
*  Neural Network Design
  Use a shallow neural network with: /n
  Input layer: One-hot encoded vector representing the current word.
  Hidden layer: Dense layer with a smaller dimension (embedding size).
  Output layer: Softmax layer to predict probabilities of words.

meow meow
```
