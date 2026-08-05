import os
from PIL import Image
import torch
import warnings

# Suppress annoying huggingface warnings
warnings.filterwarnings("ignore")


def test_florence():
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("Please install transformers: uv pip install transformers")
        return

    print("Downloading/Loading Florence-2 Base Model (this may take a minute)...")

    # Monkeypatch for a known bug in newer transformers versions with Florence-2
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
        PreTrainedTokenizerBase.additional_special_tokens = []

    # We use the base model because it is very fast on CPU compared to the large one
    model_id = "microsoft/Florence-2-base"

    # Florence-2 requires trust_remote_code=True
    # Passing attn_implementation="eager" bypasses SDPA checks that crash in newer transformers
    # We load explicitly in float32 because the base weights are mixed/half precision which crashes on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).eval()

    # Monkeypatch for the Cache object bug
    orig_prepare = model.language_model.prepare_inputs_for_generation

    def custom_prepare(*args, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
            kwargs["past_key_values"] = past_key_values.to_legacy_cache()
        return orig_prepare(*args, **kwargs)

    model.language_model.prepare_inputs_for_generation = custom_prepare

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    image_path = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\my sample\original_prescription_1.jpg"
    image = Image.open(image_path).convert("RGB")
    # Florence-2 expects square feature maps, so we resize the image to a square
    # (The processor in this transformers version fails to do this automatically)
    image = image.resize((768, 768))

    # We ask Florence-2 to perform OCR with Region (bounding boxes + text)
    prompt = "<OCR>"

    print(f"Analyzing {image_path} with Florence-2...")
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
            use_cache=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process the output to be readable
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )

    print("\n--- Florence-2 Output ---")

    # Save the output to a file to avoid Windows UnicodeEncodeErrors when printing
    output_path = "florence_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(parsed_answer))

    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    # Ensure dependencies are installed
    try:
        import einops
        import timm
    except ImportError:
        print("Installing required dependencies (einops, timm)...")
        os.system("uv pip install einops timm")
        print("Dependencies installed. Please run this script again!")
        exit(0)

    test_florence()
