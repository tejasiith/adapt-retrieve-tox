import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from . import config
from .utils import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _loader(df, tok, shuffle):
    enc = tok(list(df.smiles), padding=True, truncation=True,
              max_length=config.MAX_LEN, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(df.label.values))
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)

@torch.no_grad()
def _predict(model, loader):
    model.eval()
    out = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
        out.append(torch.softmax(logits, -1)[:, 1].cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def _embed(model, loader):
    """Mean-pooled last hidden state of the (fine-tuned) encoder."""
    model.eval()
    out = []
    for ids, mask, _ in loader:
        res = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV),
                    output_hidden_states=True)
        h = res.hidden_states[-1]
        m = mask.unsqueeze(-1).to(DEV)
        out.append(((h * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(out).numpy()

def run(mode, train_df, valid_df, test_df, seed, lora_r=8, with_embeddings=False):
    """mode: 'lora' or 'full'. Model selection uses the validation set only."""
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)

    if mode == "lora":
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=2 * lora_r,
                         lora_dropout=0.1, target_modules=["query", "value"])
        model = get_peft_model(model, cfg)   # classifier head stays trainable
        lr = config.LR_LORA
    else:
        lr = config.LR_FULL
    model.to(DEV)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{mode} seed {seed}] trainable params: {n_trainable:,} / {n_total:,}")

    train_dl = _loader(train_df, tok, shuffle=True)
    train_eval_dl = _loader(train_df, tok, shuffle=False)
    valid_dl = _loader(valid_df, tok, shuffle=False)
    test_dl = _loader(test_df, tok, shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    best_auc, best_state = -1.0, None
    for ep in range(config.EPOCHS):
        model.train()
        for ids, mask, y in train_dl:
            opt.zero_grad()
            logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
            loss = loss_fn(logits, y.to(DEV))
            loss.backward()
            opt.step()
        val_auc = roc_auc_score(valid_df.label.values, _predict(model, valid_dl))
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)       # go back to the best epoch
    result = {"test_prob": _predict(model, test_dl),
              "best_valid_auroc": best_auc,
              "n_trainable": n_trainable}
    if with_embeddings:
        result["emb_train"] = _embed(model, train_eval_dl)
        result["emb_test"] = _embed(model, test_dl)
    return result