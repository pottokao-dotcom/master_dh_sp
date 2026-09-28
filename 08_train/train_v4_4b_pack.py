#!/usr/bin/env python
# v4 版(2026-09-27):複製自 v3 打包版,只改兩處 ——
#   (1) 打包改成同卡同版面的樣本一包(v4 各卡 system 不同,隨機打包會讓共同前綴幾乎為零)
#   (2) 資料已切好 train/heldout(07_teacher/to_dataset.py),這裡不再自己切
#   超參數、loss、遮罩、前綴規則一律不動。
"""八股 LoRA —— Qwen3-4B 學生(Z-Image / FLUX.2 klein 4B 共用的文字編碼器)。blackhole 雙卡 DDP。

資料與超參數沿用 v3(train_lora_fsdp.py):r32 / alpha64 / dropout0.05 / lr1e-4 cosine / 有效批次16 / 2 epoch。
和 8B 版的差別:
- 基底是 HF 原版 Qwen/Qwen3-4B。ComfyUI 的 qwen_3_4b.safetensors 只有第 35 層 MLP up/down 不同,
  Z-Image 取第 34 層(layer_idx=-2)、klein 取 9/18/27 層,都用不到第 35 層 → 編碼端逐位元相同(2026-09-26 驗過)
- Qwen3 是混合思考模型:prompt 用 enable_thinking=False(帶空 <think></think>),與推論關思考時一致
- DDP 每卡一整份 + 只算答案位置的 logits(數學同整段算,峰值省 ~5G;見 ~/lora_pe/v3/speedup_probe)
- 4B 放得下 bf16 基底,不必 nf4(QUANT=nf4 可切回 QLoRA)
- ★共用前綴打包(使用者 2026-09-26 選定 ×4):8,830 筆共用同一段 3,370 token 的 system prompt。
  每張卡一次拿 PACK 筆,排成 [前綴][分支1]…[分支PACK],flex_attention BlockMask 讓分支只看前綴+自己,
  位置編號每個分支都從前綴尾端接續 → 每個答案 token 看到的上下文與不打包時完全相同。
  與不打包的唯一差別:前綴上的 LoRA dropout 由同一包共用。梯度實測與不打包 cos 0.9996(bf16 雜訊範圍)
- loss 正規化與 Trainer 原版相同:全步(兩卡×累積)答案 token 的 CE 總和 ÷ 全步答案 token 數

用法: accelerate launch --config_file ddp2.yaml train_lora_4b_pack.py [epochs]
"""
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainerCallback, TrainingArguments)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

BASE = Path(os.environ.get("BASE") or Path.home() / "qwen3_4b_hf")
DATA = Path(os.environ["DATA_PATH"])
OUT = Path(os.environ["OUT_DIR"])
MAXLEN = int(os.environ.get("MAXLEN", 4400))
QUANT = os.environ.get("QUANT", "bf16")
PACK = int(os.environ.get("PACK", 4))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WORLD = int(os.environ.get("WORLD_SIZE", 1))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))


class Chat(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        prompt = self.tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        answer = msgs[-1]["content"] + (self.tok.eos_token or "")
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        assert len(p_ids) + len(a_ids) <= MAXLEN, "樣本 {} 超過 MAXLEN".format(i)
        return {"p": p_ids, "a": a_ids}


class PackedByCard(Dataset):
    """v4:同卡同版面的樣本組成一包(順序固定種子打亂),每個 item 回傳一包 PACK 筆。"""
    def __init__(self, chat, groups, train):
        import random
        idx = {id(r): i for i, r in enumerate(train)}
        rng, self.chat, self.packs = random.Random(20260927), chat, []
        for key in sorted(groups):
            ids = [idx[id(r)] for r in groups[key]]
            rng.shuffle(ids)
            self.packs += [ids[k:k + PACK] for k in range(0, len(ids), PACK)]
        rng.shuffle(self.packs)

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, i):
        return [self.chat[j] for j in self.packs[i]]


def pack(batch):
    """[共同前綴][分支1]…:分支 = 該筆 prompt 剩下的部分(user 題目 + assistant 開頭)+ 答案。"""
    P = batch[0]["p"]
    n = len(P)
    for b in batch[1:]:
        n = next((k for k in range(min(n, len(b["p"]))) if b["p"][k] != P[k]), min(n, len(b["p"])))
    n = min(n, min(len(b["p"]) for b in batch) - 1)   # 每個分支至少留 prompt 最後一個 token(預測答案第一字用)
    ids, pos, seg, keep, tgt = list(P[:n]), list(range(n)), [0] * n, [], []
    for j, b in enumerate(batch, 1):
        br = b["p"][n:] + b["a"]
        start = len(ids)
        ids += br
        pos += range(n, n + len(br))
        seg += [j] * len(br)
        q = start + len(b["p"]) - n                   # 答案第一個 token 的位置
        keep += range(q - 1, q + len(b["a"]) - 1)     # 位置 t 預測 t+1
        tgt += b["a"]
    return {"input_ids": torch.tensor([ids]), "position_ids": torch.tensor([pos]),
            "seg": torch.tensor(seg), "keep": torch.tensor(keep),
            "shift_labels": torch.tensor([tgt])}       # Trainer 用它數 num_items_in_batch


class PackTrainer(Trainer):
    """forward 仍走 DDP 包好的整個模型(DDP 的梯度同步要靠它),只要 1 個位置的 logits;
    最後一層 norm 的輸出用 hook 接出來,lm_head + CE 分塊並 checkpoint,不同時持有全部答案的 logits。"""
    _h = None

    def _hook(self, mod, inp, out):
        PackTrainer._h = out

    def _get_num_items_in_batch(self, batch_samples, device):
        # Trainer 只在有 "labels" 時才數;這裡自己數:全步(累積 × 兩卡)的答案 token 總數,同原版
        n = torch.tensor(sum(b["shift_labels"].numel() for b in batch_samples), device=device)
        return self.accelerator.gather(n).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert num_items_in_batch is not None
        inner = self.model.base_model.model            # Qwen3ForCausalLM(LoRA 已注入)
        if not getattr(self, "_hooked", False):
            inner.model.norm.register_forward_hook(self._hook)
            self._hooked = True
        seg, keep, tgt = inputs["seg"], inputs["keep"], inputs["shift_labels"][0]
        L = seg.shape[0]

        def mm(b, h, qi, ki):
            return (ki <= qi) & ((seg[qi] == seg[ki]) | (seg[ki] == 0))
        bm = create_block_mask(mm, 1, None, L, L, device=seg.device)
        model(input_ids=inputs["input_ids"], position_ids=inputs["position_ids"], attention_mask=bm,
              logits_to_keep=1)
        h = PackTrainer._h[0, keep]
        PackTrainer._h = None
        loss = 0.0
        for i in range(0, h.shape[0], 512):
            loss = loss + checkpoint(lambda hh, tt: F.cross_entropy(inner.lm_head(hh).float(), tt, reduction="sum"),
                                     h[i:i + 512], tgt[i:i + 512], use_reentrant=False)
        # 與 Trainer 原版一致:÷ 全步 token 數,× 卡數(抵銷 DDP 的平均)
        return loss / num_items_in_batch * self.accelerator.num_processes


class FreeBeforeSave(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if args.save_steps and state.global_step % args.save_steps == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()


class PeakMem(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None and "loss" in logs:
            print("PEAK rank{} step {} 峰值配置 {:.2f} GiB 保留 {:.2f} GiB".format(
                os.environ.get("RANK"), state.global_step, torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30), flush=True)


def main():
    torch.cuda.set_device(LOCAL)
    epochs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    by_card = {}
    for r in rows:
        by_card.setdefault((r["card"], r.get("design", "none")), []).append(r)
    held, train = [], rows          # v4:heldout 已另存(heldout_v4_*.jsonl)
    print("訓練 {} 筆 · held-out {} 筆 · {} 組 · DDP world={} · 基底 {} ({})".format(
        len(train), len(held), len(by_card), WORLD, BASE.name, QUANT), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map={"": LOCAL})
    if QUANT == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    model.config._attn_implementation = "flex_attention"   # 樹狀遮罩要 flex;BlockMask 只算有用的區塊
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    rank = int(os.environ.get("LORA_R", 32))
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()

    ds = PackedByCard(Chat(train, tok), by_card, train)
    accum = max(1, 16 // (WORLD * PACK))   # 有效批次維持 16 = 卡數 × 每包筆數 × 累積
    print("預估 {} 個 optimizer step".format(math.ceil(len(train) * epochs / 16)), flush=True)
    args = TrainingArguments(
        output_dir=str(OUT), num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=accum,   # 1 個 item = 1 包(PACK 筆)
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=int(os.environ.get("LOG_STEPS", 10)), save_strategy="steps", save_steps=int(os.environ.get("SAVE_STEPS", 100)),
        save_total_limit=None, max_steps=int(os.environ.get("MAX_STEPS", -1)),   # v4.1:檢查點全留(使用者要比最後三個,同 8B),不影響訓練
        bf16=True, optim="adamw_torch", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, report_to=[], dataloader_num_workers=0,
        remove_unused_columns=False, ddp_find_unused_parameters=False,
    )
    print("每包 {} 筆 · 每卡累積 {} 包".format(PACK, accum), flush=True)
    trainer = PackTrainer(model=model, args=args, train_dataset=ds, data_collator=lambda b: pack(b[0]))
    trainer.add_callback(FreeBeforeSave())
    trainer.add_callback(PeakMem())
    # 只接續完整的檢查點(有 trainer_state.json);殘缺的刪掉
    import shutil
    if int(os.environ.get("RANK", 0)) == 0:
        for ck in OUT.glob("checkpoint-*"):
            if not (ck / "trainer_state.json").exists():
                print("刪除殘缺檢查點", ck.name, flush=True)
                shutil.rmtree(ck, ignore_errors=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    resume = any((ck / "trainer_state.json").exists() for ck in OUT.glob("checkpoint-*"))
    if resume:
        print("從檢查點接續", flush=True)
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(OUT / "final"))
    if trainer.is_world_process_zero():
        tok.save_pretrained(str(OUT / "final"))
        print("adapter -> {}".format(OUT / "final"), flush=True)


if __name__ == "__main__":
    main()
