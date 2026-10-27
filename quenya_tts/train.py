"""
Fine-tunes the pretrained multilingual ToucanTTS model on Quenya.

Transcripts are converted to phonemes with quenya_g2p and given to Toucan as phoneme input, with the
Finnish language embedding, so Toucan itself is used unmodified. The public model is trained only on the
volunteer recordings, which volunteers gave for training; --experimental also uses Atanquesta and Glaemscrafu,
whose licences do not allow publishing the result. The model folder gets a training.json saying which corpora
were used; quenya_tts.publish refuses to release a model trained on anything but the public corpora. Toucan's progress plots show its Finnish example sentence, since it has
no Quenya one.

Run from the repository root, on a machine with a CUDA GPU:
    python -m quenya_tts.train --gpu_id 0
    python -m quenya_tts.train --gpu_id 0 --experimental
    python -m quenya_tts.train --gpu_id 0 --corpora Glaemscrafu --steps 500   # a short test run
"""

import argparse
import json
import os
import time


def write_training_info(save_dir, corpora, public):
    """Records in the model folder which corpora the model was trained on (all of them, over resumed runs)."""
    path = os.path.join(save_dir, "training.json")
    info = {"corpora": [], "public": True, "runs": []}
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            info = json.load(f)
    info["corpora"] = sorted(set(info["corpora"]) | set(corpora))
    info["public"] = info["public"] and public
    info["runs"].append({"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "corpora": list(corpora)})
    with open(path, "w", encoding="utf8") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ToucanTTS on Quenya.")
    parser.add_argument("--gpu_id", default="0", help="which CUDA GPU to use")
    parser.add_argument("--experimental", action="store_true", help="also train on Atanquesta and Glaemscrafu (not for publishing)")
    parser.add_argument("--corpora", nargs="+", default=None,
                        help="train on exactly these corpora instead (e.g. --corpora Glaemscrafu for a test run)")
    parser.add_argument("--resume_checkpoint", default=None, help="continue from this checkpoint instead of the pretrained model")
    parser.add_argument("--resume", action="store_true", help="resume the previous run in the model folder")
    parser.add_argument("--model_dir", default=None, help="where to save checkpoints (default: Models/ToucanTTS_Quenya[_experimental])")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoint_every", type=int, default=250,
                        help="save a checkpoint (and best.pt) every this many steps; Toucan's default is once per epoch, "
                             "which on a small corpus means every few steps")
    parser.add_argument("--batch_size", type=int, default=12, help="lower this if the GPU runs out of memory")
    parser.add_argument("--lr", type=float, default=1e-5, help="with more than ~1000 recordings this can go up to 1e-4")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from quenya_tts.corpora import EXPERIMENTAL_CORPORA, PUBLIC_CORPORA, read_phonemized
    from quenya_tts.toucan import MODEL_DIR, PROXY_LANGUAGE, QuenyaPhonemizer, setup_toucan

    public = not args.experimental and set(args.corpora or PUBLIC_CORPORA) <= set(PUBLIC_CORPORA)
    save_dir = os.path.abspath(args.model_dir) if args.model_dir else os.path.join(
        MODEL_DIR, "ToucanTTS_Quenya" + ("" if public else "_experimental"))
    resume_checkpoint = os.path.abspath(args.resume_checkpoint) if args.resume_checkpoint else None
    corpora = args.corpora or PUBLIC_CORPORA + (EXPERIMENTAL_CORPORA if args.experimental else [])
    phonemize = QuenyaPhonemizer()
    transcripts = {name: read_phonemized(name, phonemize) for name in corpora}  # before changing directory

    setup_toucan(change_dir=True)
    import torch
    from huggingface_hub import hf_hub_download
    from torch.utils.data import ConcatDataset

    from Modules.ToucanTTS.ToucanTTS import ToucanTTS
    from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
    from Utility.corpus_preparation import prepare_tts_corpus
    from Utility.storage_config import MODEL_DIR as TOUCAN_MODEL_DIR
    from Utility.storage_config import PREPROCESSING_DIR

    device = torch.device("cuda")
    os.makedirs(save_dir, exist_ok=True)
    write_training_info(save_dir, corpora, public)

    datasets = [prepare_tts_corpus(transcript_dict=transcripts[name],
                                   corpus_dir=os.path.join(PREPROCESSING_DIR, "Quenya_" + name),
                                   lang=PROXY_LANGUAGE,
                                   phone_input=True)
                for name in corpora]
    train_data = ConcatDataset(datasets)

    if args.wandb:
        import wandb
        wandb.init(name=f"ToucanTTS_Quenya_{time.strftime('%Y%m%d-%H%M%S')}")

    pretrained = hf_hub_download(cache_dir=TOUCAN_MODEL_DIR, repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt")
    train_loop(net=ToucanTTS(),
               datasets=[train_data],
               device=device,
               save_directory=save_dir,
               batch_size=args.batch_size,
               eval_lang=PROXY_LANGUAGE,
               warmup_steps=500,
               lr=args.lr,
               path_to_checkpoint=resume_checkpoint or pretrained,
               fine_tune=not args.resume,
               resume=args.resume,
               steps=args.steps,
               steps_per_checkpoint=args.checkpoint_every,
               use_wandb=args.wandb,
               train_samplers=[torch.utils.data.RandomSampler(train_data)],
               gpu_count=1)
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
