"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""



import torch
import torch.nn as nn
import torch.nn.functional as F      
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.vocab_size = vocab_size

        # KLDivLoss expects log-probabilities as input
        # reduction='sum' so we can normalise by non-pad tokens ourselves
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        smooth_target = torch.zeros_like(logits)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 1))

        # Step 2: Put (1 - ε) on the correct token position
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Step 3: PAD positions get zero probability everywhere
        # We don't want any loss signal from padding tokens
        smooth_target[:, self.pad_idx] = 0

        # Step 4: Zero out rows where target IS pad
        # (these positions contribute nothing to learning)
        pad_mask = (target == self.pad_idx)
        smooth_target[pad_mask] = 0.0

        # Step 5: KLDiv needs log-softmax predictions
        log_pred = F.log_softmax(logits, dim=-1)

        # Step 6: Compute loss, normalise by number of non-pad tokens
        loss = self.criterion(log_pred, smooth_target)
        non_pad_count = (~pad_mask).sum().float()
        return loss / non_pad_count


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    import wandb

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss   = 0.0
    total_tokens = 0
    total_acc    = 0.0
    global_step  = getattr(run_epoch, 'global_step', 0)

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # ── Teacher forcing split ────────────────────────────────
            tgt_input  = tgt[:, :-1]   # [batch, tgt_len-1]
            tgt_target = tgt[:, 1:]    # [batch, tgt_len-1]

            # ── Build masks ──────────────────────────────────────────
            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            # ── Forward pass ─────────────────────────────────────────
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # ── Reshape for loss ─────────────────────────────────────
            vocab_size = logits.size(-1)
            logits_2d  = logits.reshape(-1, vocab_size)
            target_1d  = tgt_target.reshape(-1)

            # 1. Calculate Loss
            loss = loss_fn(logits_2d, target_1d)

            # 2. Calculate Token-Level Accuracy
            with torch.no_grad():
                preds = logits_2d.argmax(dim=-1)
                non_pad_mask = (target_1d != loss_fn.pad_idx)
                correct = (preds == target_1d) & non_pad_mask
                num_non_pad = non_pad_mask.sum().float()
                acc = (correct.sum().float() / torch.clamp(num_non_pad, min=1.0)).item()

            if is_train:
                optimizer.zero_grad()
                loss.backward() # <--- BACKWARD CALLED ONLY ONCE HERE

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                run_epoch.global_step = global_step

                # ── W&B logging ──────────────────────────────────────
                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/loss': loss.item(),
                    'train/accuracy': acc,
                    'train/lr'  : current_lr,
                    'epoch'     : epoch_num,
                    'step'      : global_step,
                }

                # Experiment 2.2: log Q and K gradient norms
                if global_step <= 1000:
                    q_grad_norms, k_grad_norms = [], []
                    for layer in model.encoder.layers:
                        if layer.self_attn.W_Q.weight.grad is not None:
                            q_grad_norms.append(layer.self_attn.W_Q.weight.grad.norm().item())
                        if layer.self_attn.W_K.weight.grad is not None:
                            k_grad_norms.append(layer.self_attn.W_K.weight.grad.norm().item())
                    if q_grad_norms:
                        log_dict['grad/Q_norm_mean'] = sum(q_grad_norms) / len(q_grad_norms)
                    if k_grad_norms:
                        log_dict['grad/K_norm_mean'] = sum(k_grad_norms) / len(k_grad_norms)

                # Experiment 2.5: prediction confidence
                with torch.no_grad():
                    probs = F.softmax(logits_2d, dim=-1)
                    if non_pad_mask.sum() > 0:
                        correct_probs = probs[torch.arange(len(target_1d)), target_1d][non_pad_mask]
                        log_dict['train/prediction_confidence'] = correct_probs.mean().item()

                wandb.log(log_dict)

            # Track loss and accuracy for epoch average
            non_pad = num_non_pad.item()
            total_loss   += loss.item() * non_pad
            total_acc    += acc * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc  = total_acc / max(total_tokens, 1)

    # Log validation metrics at end of epoch
    if not is_train:
        wandb.log({
            'val/loss'     : avg_loss,
            'val/accuracy' : avg_acc,
            'epoch'        : epoch_num,
        })

    return avg_loss

''' 
def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    import wandb

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss   = 0.0
    total_tokens = 0
    total_acc    = 0.0
    global_step  = getattr(run_epoch, 'global_step', 0)  # persist across calls

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # ── Teacher forcing split ────────────────────────────────
            # Input  to decoder: SOS + all tokens except last
            # Target for loss  : all tokens except SOS
            tgt_input  = tgt[:, :-1]   # [batch, tgt_len-1]
            tgt_target = tgt[:, 1:]    # [batch, tgt_len-1]

            # ── Build masks ──────────────────────────────────────────
            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            # ── Forward pass ─────────────────────────────────────────
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # ── Reshape for loss ─────────────────────────────────────
            # LabelSmoothingLoss expects [batch*tgt_len, vocab_size]
            vocab_size = logits.size(-1)
            logits_2d  = logits.reshape(-1, vocab_size)
            target_1d  = tgt_target.reshape(-1)

            loss = loss_fn(logits_2d, target_1d)

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping — prevents exploding gradients
                # Common practice for Transformers
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                run_epoch.global_step = global_step

                # ── W&B logging (training only) ──────────────────────
                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/loss': loss.item(),
                    'train/lr'  : current_lr,
                    'epoch'     : epoch_num,
                    'step'      : global_step,
                }

                # Experiment 2.2: log Q and K gradient norms
                # Only during first 1000 steps (expensive to do always)
                if global_step <= 1000:
                    q_grad_norms, k_grad_norms = [], []
                    for layer in model.encoder.layers:
                        if layer.self_attn.W_Q.weight.grad is not None:
                            q_grad_norms.append(
                                layer.self_attn.W_Q.weight.grad.norm().item()
                            )
                        if layer.self_attn.W_K.weight.grad is not None:
                            k_grad_norms.append(
                                layer.self_attn.W_K.weight.grad.norm().item()
                            )
                    if q_grad_norms:
                        log_dict['grad/Q_norm_mean'] = sum(q_grad_norms) / len(q_grad_norms)
                    if k_grad_norms:
                        log_dict['grad/K_norm_mean'] = sum(k_grad_norms) / len(k_grad_norms)

                # Experiment 2.5: prediction confidence
                # Softmax prob of the correct token (averaged over non-pad)
                # ── Reshape for loss ─────────────────────────────────────
            # LabelSmoothingLoss expects [batch*tgt_len, vocab_size]
            vocab_size = logits.size(-1)
            logits_2d  = logits.reshape(-1, vocab_size)
            target_1d  = tgt_target.reshape(-1)

            loss = loss_fn(logits_2d, target_1d)

            # ── NEW: Calculate Token-Level Accuracy ──────────────────
            with torch.no_grad():
                preds = logits_2d.argmax(dim=-1)
                non_pad_mask = (target_1d != loss_fn.pad_idx)
                correct = (preds == target_1d) & non_pad_mask
                # Avoid division by zero if batch is entirely padding (rare, but safe)
                num_non_pad = non_pad_mask.sum().float()
                acc = (correct.sum().float() / torch.clamp(num_non_pad, min=1.0)).item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                run_epoch.global_step = global_step

                # ── W&B logging (training only) ──────────────────────
                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/loss': loss.item(),
                    'train/accuracy': acc,    # <--- ADDED HERE
                    'train/lr'  : current_lr,
                    'epoch'     : epoch_num,
                    'step'      : global_step,
                }
                
                # ... (Keep your Experiment 2.2 and 2.5 logging here) ...

                wandb.log(log_dict)

            # Track loss and accuracy for epoch average
            non_pad = num_non_pad.item()
            total_loss   += loss.item() * non_pad
            total_acc    += acc * non_pad       # <--- NEW: Track total accuracy
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc  = total_acc / max(total_tokens, 1) # <--- NEW: Average epoch accuracy

    # Log validation loss AND accuracy at end of epoch
    if not is_train:
        wandb.log({
            'val/loss'     : avg_loss,
            'val/accuracy' : avg_acc,           # <--- ADDED HERE
            'epoch'        : epoch_num,
        })

    # Return loss so the training loop can track the best model
    return avg_loss
  '''             

# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    with torch.no_grad():
        # Encode source once — reuse for every decoding step
        memory = model.encode(src, src_mask)

        # Start with SOS token
        ys = torch.tensor([[start_symbol]], dtype=torch.long).to(device)

        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)

            # Greedy: argmax over vocab at the LAST position only
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)

            if next_tok.item() == end_symbol:
                break

    return ys  # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    from evaluate import load as load_metric

    model.eval()
    bleu_metric = load_metric("bleu")

    # Special token indices — skip these in output
    SOS_IDX = 2
    EOS_IDX = 3
    PAD_IDX = 1

    all_predictions  = []   # list of predicted strings
    all_references   = []   # list of reference strings (list of list)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Process one sentence at a time for greedy decode
            # (greedy_decode works with batch=1)
            for i in range(src.size(0)):
                src_i      = src[i].unsqueeze(0)          # [1, src_len]
                src_mask_i = make_src_mask(src_i).to(device)

                # Greedy decode → [1, out_len] token indices
                pred_ids = greedy_decode(
                    model       = model,
                    src         = src_i,
                    src_mask    = src_mask_i,
                    max_len     = max_len,
                    start_symbol= SOS_IDX,
                    end_symbol  = EOS_IDX,
                    device      = device,
                ).squeeze(0).tolist()  # [out_len]

                # Reference: tgt[i] — strip SOS/EOS/PAD
                ref_ids = tgt[i].tolist()

                # Convert indices → word strings, skip special tokens
                pred_words = [
                    tgt_vocab.itos[idx]
                    for idx in pred_ids
                    if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
                ]
                ref_words = [
                    tgt_vocab.itos[idx]
                    for idx in ref_ids
                    if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
                ]

                # BLEU metric expects:
                #   predictions : list of strings  ["a dog runs"]
                #   references  : list of list of strings [["a dog runs"]]
                pred_str = ' '.join(pred_words)
                ref_str  = ' '.join(ref_words)

                all_predictions.append(pred_str)
                all_references.append([ref_str])   # wrapped in list

    # Compute corpus-level BLEU
    # The evaluate library returns score in 0-1, multiply by 100
    result = bleu_metric.compute(
        predictions = all_predictions,
        references  = all_references,
    )
    bleu_score = result['bleu'] * 100

    return bleu_score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config'        : {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'd_model'       : model.d_model,
            'N'             : len(model.encoder.layers),
            'num_heads'     : model.encoder.layers[0].self_attn.num_heads,
            'd_ff'          : model.encoder.layers[0].ffn.linear1.out_features,
            'dropout'       : 0.1,   # stored as fixed; not accessible post-init
        },
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    ckpt = torch.load(path, map_location='cpu')

    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    print(f"Checkpoint loaded from {path}, epoch {ckpt['epoch']}")
    return ckpt['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    import argparse
    import wandb
    from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
    from lr_scheduler import NoamScheduler

    # ── Argument Parser ───────────────────────────────────────────────
    # All W&B experiments controlled via CLI — no code changes needed
    parser = argparse.ArgumentParser(description="Train Transformer NMT")

    # Model hyperparameters
    parser.add_argument('--d_model',      type=int,   default=256)
    parser.add_argument('--N',            type=int,   default=3)
    parser.add_argument('--num_heads',    type=int,   default=8)
    parser.add_argument('--d_ff',         type=int,   default=512)
    parser.add_argument('--dropout',      type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--num_epochs',   type=int,   default=15)
    parser.add_argument('--warmup_steps', type=int,   default=4000)
    parser.add_argument('--lr',           type=float, default=None,
                        help='Fixed LR (overrides Noam). For experiment 2.1')

    # Label smoothing — experiment 2.5
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='0.0 = standard CE, 0.1 = label smoothing')

    # Attention scaling — experiment 2.2
    parser.add_argument('--use_scale',    type=int,   default=1,
                        help='1=use √dk scaling, 0=no scaling')

    # Positional encoding — experiment 2.4
    parser.add_argument('--pos_encoding', type=str,   default='sinusoidal',
                        choices=['sinusoidal', 'learned'],
                        help='sinusoidal (paper) or learned (experiment 2.4)')

    # W&B
    parser.add_argument('--wandb_project', type=str, default='da6401-a3')
    parser.add_argument('--run_name',      type=str, default=None)

    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_best_only', type=int, default=1)

    args = parser.parse_args()

    # ── W&B Init ─────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.run_name,
        config  = vars(args),   # log all hyperparams
    )
    config = wandb.config

    # ── Device ───────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Dataset & Dataloaders ─────────────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=config.batch_size, num_workers=2)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Src vocab: {src_vocab_size}, Tgt vocab: {tgt_vocab_size}")

    wandb.config.update({
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
    })

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = config.d_model,
        N              = config.N,
        num_heads      = config.num_heads,
        d_ff           = config.d_ff,
        dropout        = config.dropout,
    ).to(device)

    # Experiment 2.2: disable scaling in attention if requested
    # We patch scaled_dot_product_attention via a flag on the model
    if config.use_scale == 0:
        import model as model_module
        _orig_sdpa = model_module.scaled_dot_product_attention
        def _no_scale_sdpa(Q, K, V, mask=None):
            # Same as original but WITHOUT the √dk division
            import math, torch.nn.functional as F
            scores = torch.matmul(Q, K.transpose(-2, -1))  # no scaling!
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            attn_w = F.softmax(scores, dim=-1)
            return torch.matmul(attn_w, V), attn_w
        model_module.scaled_dot_product_attention = _no_scale_sdpa
        print("WARNING: Attention scaling DISABLED (experiment 2.2)")

    # Experiment 2.4: replace sinusoidal PE with learned embeddings
    if config.pos_encoding == 'learned':
        import torch.nn as nn
        max_len = 5000
        model.pos_encoding = nn.Sequential(
            # Learned positional embedding table
            # We wrap it so forward(x) still works like PE
        )
        # Simpler: replace with a LearnedPE module
        class LearnedPE(nn.Module):
            def __init__(self, d_model, dropout, max_len=5000):
                super().__init__()
                self.dropout   = nn.Dropout(p=dropout)
                self.pos_embed = nn.Embedding(max_len, d_model)
            def forward(self, x):
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                return self.dropout(x + self.pos_embed(positions))
        model.pos_encoding = LearnedPE(
            config.d_model, config.dropout
        ).to(device)
        print("Using LEARNED positional encoding (experiment 2.4)")

    # Log model parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({'n_params': n_params})

    # ── Optimizer ────────────────────────────────────────────────────
    # Paper: Adam with β1=0.9, β2=0.98, ε=1e-9
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr   = 1.0,       # base LR — Noam scheduler scales this
        betas= (0.9, 0.98),
        eps  = 1e-9,
    )

    # ── Scheduler ────────────────────────────────────────────────────
    # Experiment 2.1: fixed LR vs Noam
    if config.lr is not None:
        # Fixed LR baseline for experiment 2.1
        for pg in optimizer.param_groups:
            pg['lr'] = config.lr
        scheduler = None
        print(f"Using FIXED LR: {config.lr} (experiment 2.1 baseline)")
    else:
        scheduler = NoamScheduler(
            optimizer,
            d_model      = config.d_model,
            warmup_steps = config.warmup_steps,
        )
        print(f"Using Noam scheduler (warmup={config.warmup_steps})")

    # ── Loss ─────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = tgt_vocab_size,
        pad_idx    = PAD_IDX,
        smoothing  = config.label_smoothing,
    )

    # ── Checkpoint dir ───────────────────────────────────────────────
    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_ckpt_path = os.path.join(config.checkpoint_dir, 'best_model.pt')

    # ── Training Loop ─────────────────────────────────────────────────
    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")

        # Train
        train_loss = run_epoch(
            data_iter = train_loader,
            model     = model,
            loss_fn   = loss_fn,
            optimizer = optimizer,
            scheduler = scheduler,
            epoch_num = epoch,
            is_train  = True,
            device    = device,
        )

        # Validate
        val_loss = run_epoch(
            data_iter = val_loader,
            model     = model,
            loss_fn   = loss_fn,
            optimizer = None,
            scheduler = None,
            epoch_num = epoch,
            is_train  = False,
            device    = device,
        )
        

        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # W&B: log epoch-level metrics
        wandb.log({
            'epoch/train_loss': train_loss,
            'epoch/val_loss'  : val_loss,
            'epoch'           : epoch,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler or
                            torch.optim.Adam(model.parameters()),
                            epoch, best_ckpt_path)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_epoch']    = epoch

        # Also save every epoch checkpoint
        epoch_ckpt = os.path.join(
            config.checkpoint_dir, f'epoch_{epoch+1}.pt'
        )
        save_checkpoint(model, optimizer,
                        scheduler or torch.optim.Adam(model.parameters()),
                        epoch, epoch_ckpt)

   # ── Experiment 2.3: Attention heatmaps ───────────────────────────
    print("\nLogging attention heatmaps...")
    _log_attention_heatmaps(model, val_loader, src_vocab, tgt_vocab, device)

    # ── Final Evaluations (Run Once!) ────────────────────────────
    # MUST load the best checkpoint BEFORE evaluating!
    print("\nLoading best checkpoint for final evaluation...")
    load_checkpoint(best_ckpt_path, model)
    model.eval()

    print("Evaluating final Validation BLEU...")
    val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device)
    wandb.run.summary['final_val_bleu'] = val_bleu
    print(f"Validation BLEU: {val_bleu:.2f}")

    print("Evaluating final Test BLEU...")
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
    wandb.run.summary['final_test_bleu'] = test_bleu
    print(f"Test BLEU: {test_bleu:.2f}")

    wandb.finish()

def _log_attention_heatmaps(model, val_loader, src_vocab, tgt_vocab, device):
    """
    Experiment 2.3: Log attention heatmap for each head in last encoder layer.
    Picks one sample from val set and visualizes all heads.
    """
    import wandb
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for servers

    model.eval()
    with torch.no_grad():
        # Get one batch, pick first sentence
        src, tgt = next(iter(val_loader))
        src = src[:1].to(device)   # [1, src_len]

        src_mask = make_src_mask(src).to(device)

        # Forward through encoder to populate attn_w in each layer
        _ = model.encode(src, src_mask)

        # Get attention weights from LAST encoder layer
        last_layer = model.encoder.layers[-1]
        attn_w = last_layer.self_attn.attn_w  # [1, num_heads, src_len, src_len]

        # Convert src token indices → words for axis labels
        src_tokens = [
            src_vocab.itos.get(idx.item(), '<unk>')
            for idx in src[0]
            if idx.item() != 1   # skip PAD
        ]

        num_heads = attn_w.size(1)
        fig, axes = plt.subplots(2, num_heads // 2,
                                  figsize=(4 * num_heads // 2, 8))
        axes = axes.flatten()

        for h in range(num_heads):
            head_attn = attn_w[0, h].cpu().numpy()
            # Trim to non-pad length
            n = len(src_tokens)
            head_attn = head_attn[:n, :n]

            ax = axes[h]
            im = ax.imshow(head_attn, cmap='Blues', aspect='auto')
            ax.set_title(f'Head {h+1}')
            ax.set_xticks(range(n))
            ax.set_xticklabels(src_tokens, rotation=90, fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(src_tokens, fontsize=7)

        plt.suptitle('Last Encoder Layer — Attention Heads', fontsize=14)
        plt.tight_layout()

        # Log to W&B
        wandb.log({'attention/encoder_last_layer': wandb.Image(fig)})
        plt.close(fig)
        print("Attention heatmaps logged to W&B")



if __name__ == "__main__":
    run_training_experiment() 

''' 
if __name__ == "__main__":
    loss_fn = LabelSmoothingLoss(vocab_size=100, pad_idx=1, smoothing=0.1)

    logits = torch.randn(8, 100)        # 8 tokens, 100 vocab
    target = torch.tensor([5,2,7,1,3,9,1,4])  # 1s are PAD

    loss = loss_fn(logits, target)
    print(loss)           # scalar, reasonable positive value
    print(loss.shape)     # torch.Size([])  — scalar
    from model import Transformer, make_src_mask, make_tgt_mask

    model    = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=800,
        d_model=128, N=2, num_heads=4, d_ff=256, dropout=0.0
    )
    src      = torch.randint(2, 1000, (1, 10))
    src_mask = make_src_mask(src)

    out = greedy_decode(
        model=model, src=src, src_mask=src_mask,
        max_len=20, start_symbol=2, end_symbol=3, device='cpu'
    )
    print(out.shape)        # [1, something up to 21]
    print(out[0, 0].item()) # should be 2 (SOS)
    '''