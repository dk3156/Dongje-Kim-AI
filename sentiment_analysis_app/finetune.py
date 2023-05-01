import torch
import transformers
#===================================#
"""
Documentation for Milestone 4 - Dongje Kim, dk3156
Incline comments are for the detailed documentation of each part of the code
"""
#===================================#
"""
URL for downloading the pre-trained models.
"""
DOWNLOAD_URL = "https://github.com/unitaryai/detoxify/releases/download/"
MODEL_URLS = {
    "original": DOWNLOAD_URL + "v0.1-alpha/toxic_original-c1212f89.ckpt",
}

PRETRAINED_MODEL = None

"""
Returns: a tuple containing the pre-trained model and its corresponding tokenizer
Arguments:
model_type: the type of pre-trained model to load.
model_name: the name of the pre-trained model class.
tokenizer_name: the name of the tokenizer class.
num_classes: the number of classes that the model should classify the input into.
state_dict: the state dictionary of the pre-trained model.
huggingface_config_path: the path to the Hugging Face configuration file.
"""
def get_model_and_tokenizer(
    model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None
):
    model_class = getattr(transformers, model_name)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=huggingface_config_path or model_type,
        num_labels=num_classes,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
    )

    return model, tokenizer

"""
Returns : loads the pre-trained model and its tokenizer, and returns them along with the class names that the model classifies the input into.
Arguments:
model_type: the type of pre-trained model to load.
checkpoint: the path to the checkpoint file.
device: the device to run the model on.
huggingface_config_path: the path to the Hugging Face configuration file.
"""
def load_checkpoint(model_type="original", checkpoint=None, device="cpu", huggingface_config_path=None):
    if checkpoint is None:
        checkpoint_path = MODEL_URLS[model_type]
        loaded = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
    else:
        loaded = torch.load(checkpoint, map_location=device)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    # standardise class names between models
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model, tokenizer = get_model_and_tokenizer(
        **loaded["config"]["arch"]["args"],
        state_dict=loaded["state_dict"],
        huggingface_config_path=huggingface_config_path,
    )

    return model, tokenizer, class_names

"""
loads the pre-trained model and returns it.
"""
def load_model(model_type, checkpoint=None):
    if checkpoint is None:
        model, _, _ = load_checkpoint(model_type=model_type)
    else:
        model, _, _ = load_checkpoint(checkpoint=checkpoint)
    return model

"""
Finetune Class
    Easily predict if a comment or list of comments is toxic.
    Can initialize a model type or checkpoint path:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
    Args:
        model_type(str): model type to be loaded, can be either original,
                         unbiased or multilingual
        checkpoint(str): checkpoint path, defaults to None
        device(str or torch.device): accepts any torch.device input or
                                     torch.device object, defaults to cpu
        huggingface_config_path: path to HF config and tokenizer files needed for offline model loading
    Returns:
        results(dict): dictionary of output scores for each class
    """
class Finetune:
    def __init__(self, model_type="original", checkpoint=PRETRAINED_MODEL, device="cpu", huggingface_config_path=None):
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            huggingface_config_path=huggingface_config_path,
        )
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i] if isinstance(text, str) else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results

def toxic_bert():
    return load_model("original")
