# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from pathlib import Path

import librosa
import torch
import torchaudio
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

from funaudiochat.register import register_funaudiochat
from utils.constant import AUDIO_TEMPLATE, DEFAULT_S2M_GEN_KWARGS, DEFAULT_SP_GEN_KWARGS, SPOKEN_S2M_PROMPT

register_funaudiochat()


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _patch_hyperpyyaml_ruamel_compat():
    """Patch HyperPyYAML loader for ruamel.yaml API mismatch on some remote envs."""
    try:
        from hyperpyyaml.core import Loader
        if not hasattr(Loader, "max_depth"):
            Loader.max_depth = None
    except Exception:
        # Keep inference path unchanged when hyperpyyaml is absent at import time.
        pass


def _load_detokenizer_utils():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidate_paths = [
        repo_root / "third_party" / "CosyVoice",
        repo_root / "third_party" / "CosyVoice" / "third_party" / "Matcha-TTS",
        Path("/home/chang168/personaplex/Fun-Audio-Chat/third_party/CosyVoice"),
        Path("/home/chang168/personaplex/Fun-Audio-Chat/third_party/CosyVoice/third_party/Matcha-TTS"),
    ]

    for p in candidate_paths:
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)

    try:
        from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
        return get_audio_detokenizer, token2wav
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import CosyVoice detokenizer modules. "
            "Please ensure CosyVoice submodule exists under "
            "~/personaplex/Fun-Audio-Chat/third_party/CosyVoice "
            "and contains package 'cosyvoice/cli'."
        ) from exc


def _build_model(model_path: str, use_kv_cache: bool = True):
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    model.config.use_cache = use_kv_cache
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = use_kv_cache

    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs["text_greedy"] = True

    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["max_new_tokens"] = 2048
    gen_kwargs["use_cache"] = use_kv_cache

    model.sp_gen_kwargs.update(sp_gen_kwargs)

    return processor, model, gen_kwargs


def _generate_one_turn(processor, model, gen_kwargs, cosyvoice_model, conversation, audio_list, input_audio_path, output_audio_path):
    audio_list.append(librosa.load(str(input_audio_path), sr=16000)[0])
    conversation.append({"role": "user", "content": AUDIO_TEMPLATE})

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio_list, return_tensors="pt", return_token_type_ids=False).to(model.device)

    with torch.inference_mode():
        generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)

    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)

    token_for_cosyvoice = [x for x in audio_ids[0].tolist() if 0 <= x < 6561]
    speech = token2wav(
        cosyvoice_model,
        token_for_cosyvoice,
        embedding=None,
        token_hop_len=25 * 30,
        pre_lookahead_len=3,
    )

    torchaudio.save(str(output_audio_path), speech.cpu(), cosyvoice_model.sample_rate)
    conversation.append({"role": "assistant", "content": generate_text})


def infer_dataset_kv_cache(model_path: str, root_dir: str, use_kv_cache: bool = True):
    processor, model, gen_kwargs = _build_model(model_path, use_kv_cache=use_kv_cache)
    _patch_hyperpyyaml_ruamel_compat()
    get_audio_detokenizer, token2wav = _load_detokenizer_utils()
    print("Loading CosyVoice detokenizer...")
    cosyvoice_model = get_audio_detokenizer()
    print(f"KV cache enabled: {use_kv_cache}")

    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"root_dir is invalid: {root_dir}")

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not sample_dirs:
        print(f"No sample directories found under: {root_dir}")
        return

    total = 0
    success = 0
    skipped = 0
    failed = 0

    for sample_dir in sample_dirs:
        total += 1
        input_question = sample_dir / "input_question1.wav"
        input_interrupt = sample_dir / "input_question2.wav"
        output_question = sample_dir / "output_question.wav"
        output_interrupt = sample_dir / "output_interrupt.wav"

        if not input_question.exists() or not input_interrupt.exists():
            print(f"[SKIP] Missing required wav(s) in {sample_dir}")
            skipped += 1
            continue

        conversation = [{"role": "system", "content": SPOKEN_S2M_PROMPT}]
        audio_list = []

        try:
            _generate_one_turn(
                processor,
                model,
                gen_kwargs,
                cosyvoice_model,
                conversation,
                audio_list,
                input_question,
                output_question,
            )
            _generate_one_turn(
                processor,
                model,
                gen_kwargs,
                cosyvoice_model,
                conversation,
                audio_list,
                input_interrupt,
                output_interrupt,
            )
            print(f"[OK] {sample_dir}")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {sample_dir}: {exc}")
            failed += 1

    print("\nDone.")
    print(f"Total:   {total}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Batch KV-cache multiturn inference: root_dir/*/(input_question1.wav,input_question2.wav) -> output_question.wav,output_interrupt.wav"
    )
    parser.add_argument("--model-path", type=str, required=True, help="Model path, e.g. pretrained_models/Fun-Audio-Chat-8B")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing sample subdirectories")
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache. By default KV cache is enabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    infer_dataset_kv_cache(
        model_path=args.model_path,
        root_dir=args.root_dir,
        use_kv_cache=not args.no_kv_cache,
    )
