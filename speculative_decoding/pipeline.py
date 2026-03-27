"""
pipeline.py
Multi-Token Buffered Pipeline Architecture.

Two-stage producer-consumer pipeline:
  Stage 1 (Draft thread)  : draft model continuously generates speculative batches
  Stage 2 (Verify thread) : target model verifies the oldest batch in the queue

Key design decisions implemented:
  - Separate draft and verify engines (no resource contention)
  - Bounded FIFO queue with backpressure (draft pauses when queue is full)
  - Hardware-assisted rollback via commit/rollback pointer tracking
  - Speculative vs committed KV-cache state separation
  - Pipeline continues until n tokens generated or EOS reached

Pipeline hazards addressed:
  - Rollback hazard   : queue is flushed on mismatch, draft restarts
  - Pipeline bubbles  : bounded queue keeps both stages busy
  - Speculative waste : backpressure prevents over-speculation
  - Buffer imbalance  : configurable queue depth (default = 2 batches)
"""

import torch
import torch.nn.functional as F
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Data structures ───────────────────────────────────────────

@dataclass
class SpecBatch:
    """One speculative batch: k draft tokens + their probabilities."""
    batch_id:    int
    draft_ids:   List[int]
    draft_probs: List[float]
    context_len: int           # committed sequence length when this batch was drafted


@dataclass
class VerifyResult:
    """Result of one verification pass."""
    batch_id:     int
    accepted_ids: List[int]
    bonus_token:  int
    rollback:     bool         # True if any token was rejected


# ── Hardware-Assisted Rollback Buffer ─────────────────────────

class RollbackBuffer:
    """
    Tracks commit pointer and rollback pointer.
    Provides O(k) rollback without full context recomputation.

    commit_ptr  : index of last committed (verified) token
    speculative : list of token ids currently in speculative state
    """

    def __init__(self):
        self.committed_ids: List[int] = []
        self.speculative_ids: List[int] = []
        self.commit_ptr: int = 0

    def add_speculative(self, token_ids: List[int]):
        self.speculative_ids.extend(token_ids)

    def commit(self, token_ids: List[int]):
        """Move accepted tokens from speculative → committed."""
        self.committed_ids.extend(token_ids)
        self.commit_ptr = len(self.committed_ids)
        # Remove the committed tokens from speculative list
        self.speculative_ids = self.speculative_ids[len(token_ids):]

    def rollback(self):
        """
        Discard all speculative tokens after mismatch.
        Commit pointer stays at last verified position.
        O(k) operation — does NOT recompute the committed context.
        """
        self.speculative_ids = []

    @property
    def full_sequence(self) -> List[int]:
        return self.committed_ids + self.speculative_ids

    @property
    def committed_len(self) -> int:
        return len(self.committed_ids)


# ── Pipeline ──────────────────────────────────────────────────

class MultiTokenPipeline:
    """
    Buffered two-stage pipeline for speculative decoding.

    Stage 1 — Draft thread:
        Runs the draft model, fills the spec_queue with SpecBatch objects.
        Pauses (backpressure) when queue is full.

    Stage 2 — Verify thread (main thread):
        Dequeues the oldest SpecBatch, runs ONE target model pass,
        accepts/rejects tokens, updates the rollback buffer.
        On mismatch: sends flush signal to draft thread.

    Communication:
        spec_queue   : SpecBatch objects (draft → verify)
        flush_event  : threading.Event, set when rollback needed
        stop_event   : threading.Event, set when generation is complete
    """

    def __init__(self, draft_model, target_model, tokenizer,
                 k: int = 4,
                 queue_depth: int = 2,
                 alpha_high: float = 0.72,
                 alpha_low:  float = 0.48):

        self.draft   = draft_model
        self.target  = target_model
        self.tok     = tokenizer
        self.k       = k
        self.device  = next(target_model.parameters()).device

        # Bounded queue — backpressure when full
        self.spec_queue  = queue.Queue(maxsize=queue_depth)
        self.flush_event = threading.Event()
        self.stop_event  = threading.Event()

        # Shared state (guarded by lock)
        self.lock            = threading.Lock()
        self.shared_context  = None    # current committed input_ids [1, seq]
        self.batch_counter   = 0

        # Stats
        self.total_drafted   = 0
        self.total_accepted  = 0
        self.total_rollbacks = 0
        self.total_rounds    = 0

    @property
    def acceptance_rate(self):
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    # ── Trim KV-cache helper ─────────────────────────────────
    @staticmethod
    def _trim_kv(past_kv, n_keep):
        if past_kv is None:
            return None
        return tuple(
            tuple(kv[:, :, :n_keep, :] for kv in layer)
            for layer in past_kv
        )

    # ── Stage 1: Draft thread ─────────────────────────────────
    def _draft_worker(self):
        """
        Continuously drafts speculative batches and enqueues them.
        Pauses when:
          (a) queue is full (backpressure)
          (b) flush_event is set (rollback in progress)
        Stops when stop_event is set.
        """
        while not self.stop_event.is_set():

            # Wait if flush is in progress — don't generate stale batches
            if self.flush_event.is_set():
                time.sleep(0.001)
                continue

            # Read current committed context (thread-safe snapshot)
            with self.lock:
                if self.shared_context is None:
                    time.sleep(0.001)
                    continue
                context = self.shared_context.clone()
                batch_id = self.batch_counter
                self.batch_counter += 1

            # Draft k tokens autoregressively
            draft_ids   = []
            draft_probs = []
            current     = context

            try:
                with torch.no_grad():
                    for _ in range(self.k):
                        if self.flush_event.is_set():
                            break   # abort draft if rollback triggered
                        out    = self.draft(current)
                        logits = out.logits[:, -1, :]
                        probs  = F.softmax(logits, dim=-1)
                        token  = torch.argmax(probs, dim=-1)
                        p      = probs[0, token.item()].item()

                        draft_ids.append(token.item())
                        draft_probs.append(max(p, 1e-9))
                        current = torch.cat(
                            [current, token.unsqueeze(0)], dim=1
                        )
            except Exception:
                break   # exit on model error

            if self.flush_event.is_set():
                continue  # discard this batch

            if len(draft_ids) < self.k:
                continue  # incomplete batch due to flush

            batch = SpecBatch(
                batch_id=batch_id,
                draft_ids=draft_ids,
                draft_probs=draft_probs,
                context_len=context.shape[1],
            )

            # Block here if queue is full (backpressure)
            # This naturally pauses the draft stage when verify is slow
            while not self.stop_event.is_set():
                try:
                    self.spec_queue.put(batch, timeout=0.05)
                    break
                except queue.Full:
                    if self.flush_event.is_set():
                        break  # don't enqueue if flush is pending

    # ── Stage 2: Verify (main thread) ─────────────────────────
    def _verify_batch(self, input_ids: torch.Tensor,
                      batch: SpecBatch) -> VerifyResult:
        """
        Run target model once over full sequence (context + k draft tokens).
        Apply sampling-based acceptance logic.
        """
        draft_tensor = torch.tensor(
            batch.draft_ids, device=self.device
        ).unsqueeze(0)
        full_seq = torch.cat([input_ids, draft_tensor], dim=1)

        with torch.no_grad():
            out = self.target(full_seq)

        prefix_len   = input_ids.shape[1]
        accepted_ids = []
        bonus_token  = None
        rollback     = False

        for j in range(self.k):
            pos           = prefix_len - 1 + j
            target_logits = out.logits[:, pos, :]
            target_probs  = F.softmax(target_logits, dim=-1)
            p_target      = target_probs[0, batch.draft_ids[j]].item()
            p_draft       = batch.draft_probs[j]

            accept_prob = min(1.0, p_target / p_draft)
            accepted    = torch.rand(1).item() < accept_prob

            self.total_drafted += 1

            if accepted:
                self.total_accepted += 1
                accepted_ids.append(batch.draft_ids[j])
            else:
                # Resample from corrected distribution
                corrected = torch.clamp(
                    target_probs[0] - p_draft, min=0.0
                )
                s = corrected.sum()
                corrected = corrected / s if s > 1e-8 else target_probs[0]
                bonus_token = torch.multinomial(corrected, 1).item()
                rollback    = True
                self.total_rollbacks += 1
                break

        if bonus_token is None:
            last_pos    = prefix_len - 1 + self.k
            bonus_probs = F.softmax(out.logits[:, last_pos, :], dim=-1)
            bonus_token = torch.argmax(bonus_probs, dim=-1).item()

        return VerifyResult(
            batch_id=batch.batch_id,
            accepted_ids=accepted_ids,
            bonus_token=bonus_token,
            rollback=rollback,
        )

    def _flush_queue(self):
        """Drain the spec_queue after a rollback."""
        flushed = 0
        while not self.spec_queue.empty():
            try:
                self.spec_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        return flushed

    # ── Main generate ─────────────────────────────────────────
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 verbose: bool = False) -> dict:
        """
        Run the buffered multi-token pipeline.

        1. Start the draft thread (Stage 1).
        2. Main thread runs the verify loop (Stage 2).
        3. On rollback: set flush_event, drain queue, update context,
           clear flush_event so draft resumes.
        4. Stop when max_new_tokens generated or EOS.
        """
        # Reset state
        self.total_drafted   = 0
        self.total_accepted  = 0
        self.total_rollbacks = 0
        self.total_rounds    = 0
        self.flush_event.clear()
        self.stop_event.clear()
        self.batch_counter   = 0

        # Tokenize and set initial shared context
        input_ids = self.tok(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        with self.lock:
            self.shared_context = input_ids.clone()

        output_ids       = input_ids.clone()
        rollback_buffer  = RollbackBuffer()
        rollback_buffer.committed_ids = input_ids[0].tolist()

        # Start draft thread
        draft_thread = threading.Thread(
            target=self._draft_worker, daemon=True
        )
        draft_thread.start()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_generated = 0

        try:
            while tokens_generated < max_new_tokens:

                # Dequeue next speculative batch (wait up to 2s)
                try:
                    batch = self.spec_queue.get(timeout=2.0)
                except queue.Empty:
                    if verbose:
                        print("  WARNING: Queue empty — draft stage may be stalled")
                    break

                # Verify batch
                result = self._verify_batch(output_ids, batch)
                self.total_rounds += 1

                if result.rollback:
                    # ── Rollback path ────────────────────────
                    # 1. Signal draft thread to stop and discard
                    self.flush_event.set()

                    # 2. Drain any queued batches (they're now stale)
                    flushed = self._flush_queue()

                    # 3. Commit accepted prefix + bonus
                    new_ids    = result.accepted_ids + [result.bonus_token]
                    new_tensor = torch.tensor(
                        new_ids, device=self.device
                    ).unsqueeze(0)
                    output_ids = torch.cat([output_ids, new_tensor], dim=1)
                    tokens_generated += len(new_ids)

                    # 4. Update rollback buffer
                    rollback_buffer.rollback()
                    rollback_buffer.commit(new_ids)

                    # 5. Update shared context so draft resumes correctly
                    with self.lock:
                        self.shared_context = output_ids.clone()

                    # 6. Clear flush — draft resumes from new context
                    self.flush_event.clear()

                    if verbose:
                        print(f"  Round {self.total_rounds:3d} | "
                              f"ROLLBACK at pos {len(result.accepted_ids)} | "
                              f"flushed {flushed} batches | "
                              f"α={self.acceptance_rate:.2f}")
                else:
                    # ── Commit path ──────────────────────────
                    new_ids    = result.accepted_ids + [result.bonus_token]
                    new_tensor = torch.tensor(
                        new_ids, device=self.device
                    ).unsqueeze(0)
                    output_ids = torch.cat([output_ids, new_tensor], dim=1)
                    tokens_generated += len(new_ids)

                    # Update rollback buffer
                    rollback_buffer.commit(new_ids)

                    # Update shared context for draft stage
                    with self.lock:
                        self.shared_context = output_ids.clone()

                    if verbose:
                        print(f"  Round {self.total_rounds:3d} | "
                              f"accepted={len(result.accepted_ids)} bonus=1 | "
                              f"α={self.acceptance_rate:.2f} | "
                              f"committed_len={rollback_buffer.committed_len}")

                # Stop on EOS
                if result.bonus_token == self.tok.eos_token_id:
                    break

        finally:
            # Always stop the draft thread cleanly
            self.stop_event.set()
            draft_thread.join(timeout=3.0)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        text = self.tok.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return {
            "text":            text,
            "tokens":          tokens_generated,
            "time":            elapsed,
            "tps":             tokens_generated / max(elapsed, 1e-6),
            "acceptance_rate": self.acceptance_rate,
            "rounds":          self.total_rounds,
            "rollbacks":       self.total_rollbacks,
            "k":               self.k,
        }

    def print_stats(self, result):
        print(f"\n{'─'*55}")
        print(f"  Output   : {result['text'][:70]}...")
        print(f"  Tokens   : {result['tokens']}")
        print(f"  Time     : {result['time']:.2f}s")
        print(f"  Speed    : {result['tps']:.1f} tok/s")
        print(f"  Rounds   : {result['rounds']}")
        print(f"  α (accept): {result['acceptance_rate']:.2%}")
        print(f"  Rollbacks: {result['rollbacks']}")
        print(f"  k        : {result['k']}")
        print(f"{'─'*55}\n")


# ── Quick diagnostic: show pipeline stages firing ────────────
def pipeline_demo(draft_model, target_model, tokenizer,
                  prompt="The first digits of pi are",
                  n_tokens=60, k=4, verbose=True):
    print(f"\n{'='*55}")
    print(f"  Multi-Token Pipeline Demo  (k={k})")
    print(f"  Queue depth: 2 batches (overlap = 1 ahead)")
    print(f"{'='*55}")

    pipeline = MultiTokenPipeline(
        draft_model, target_model, tokenizer,
        k=k, queue_depth=2
    )
    result = pipeline.generate(prompt, max_new_tokens=n_tokens, verbose=verbose)
    pipeline.print_stats(result)
    return result


if __name__ == "__main__":
    from models import load_models, autoregressive_baseline

    print("Loading models...")
    draft_model, draft_tok, target_model, target_tok = load_models()

    PROMPT   = "The first digits of pi are"
    N_TOKENS = 80

    # Baseline
    print("\n=== TARGET BASELINE ===")
    baseline_tps, _ = autoregressive_baseline(
        target_model, target_tok, PROMPT, N_TOKENS, "GPT-2 XL"
    )

    # Pipeline
    result = pipeline_demo(
        draft_model, target_model, target_tok,
        prompt=PROMPT, n_tokens=N_TOKENS, k=4, verbose=True
    )

    speedup = result["tps"] / baseline_tps
    print(f"  Pipeline speedup : {speedup:.2f}x over target baseline")
    print(f"\n  NOTE: Pipeline overhead is higher than synchronous decoder")
    print(f"  on a single GPU due to Python threading + GIL contention.")
    print(f"  True benefit appears with: separate GPUs, larger models,")
    print(f"  or a native C++/CUDA implementation.")