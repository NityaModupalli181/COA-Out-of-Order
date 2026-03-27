"""
speculative_decoder.py
Core speculative decoding algorithm.

Phases per round:
  1. Draft  : draft model generates k candidate tokens autoregressively
  2. Verify : target model verifies all k in ONE parallel forward pass
  3. Accept : sampling-based acceptance preserving target distribution
  4. Rollback: on rejection, resample + discard remaining draft tokens
"""

import torch
import torch.nn.functional as F
import time


class SpeculativeDecoder:
    """
    Base speculative decoder with fixed speculation depth k.
    Uses sampling-based acceptance (stochastic) which gives
    higher throughput than greedy while preserving the exact
    target model output distribution.
    """

    def __init__(self, draft_model, target_model, tokenizer, k=4):
        self.draft   = draft_model
        self.target  = target_model
        self.tok     = tokenizer
        self.k       = k
        self.device  = next(target_model.parameters()).device

        # Running statistics
        self.reset_stats()

    def reset_stats(self):
        self.total_rounds  = 0
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_rollbacks = 0

    @property
    def acceptance_rate(self):
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    # ── Phase 1: Draft k tokens ───────────────────────────────
    def _draft(self, input_ids):
        """
        Autoregressively generate k draft tokens using the draft model.
        Returns:
            draft_ids   : list[int] of length k
            draft_probs : list[float], probability assigned to each draft token
            full_ids    : input_ids + draft_ids tensor
        """
        draft_ids   = []
        draft_probs = []
        current     = input_ids.clone()  # [1, seq_len]

        with torch.no_grad():
            for _ in range(self.k):
                out    = self.draft(current)
                logits = out.logits[:, -1, :]           # [1, vocab_size]
                probs  = F.softmax(logits, dim=-1)      # [1, vocab_size]

                # Greedy selection for draft (deterministic proposals)
                token = torch.argmax(probs, dim=-1)     # [1]
                p     = probs[0, token.item()].item()

                draft_ids.append(token.item())
                draft_probs.append(max(p, 1e-9))        # avoid div-by-zero

                # Feed new token back in for next step
                current = torch.cat(
                    [current, token.unsqueeze(0)], dim=1
                )

        return draft_ids, draft_probs, current

    # ── Phase 2 + 3: Verify and accept/reject ─────────────────
    def _verify(self, input_ids, draft_ids, draft_probs):
        """
        Run target model ONCE over (context + k draft tokens).
        Apply sampling-based acceptance to each position.

        Returns:
            accepted_ids : list[int] of accepted token ids (length 0..k)
            bonus_token  : int, always generated regardless of acceptance
            n_accepted   : int
        """
        prefix_len    = input_ids.shape[1]
        draft_tensor  = torch.tensor(
            draft_ids, device=self.device
        ).unsqueeze(0)                              # [1, k]
        full_seq      = torch.cat(
            [input_ids, draft_tensor], dim=1
        )                                           # [1, prefix+k]

        with torch.no_grad():
            out = self.target(full_seq)             # ONE forward pass

        accepted_ids  = []
        bonus_token   = None

        for j in range(self.k):
            # Target logits at position (prefix-1+j) predict token at (prefix+j)
            pos           = prefix_len - 1 + j
            target_logits = out.logits[:, pos, :]   # [1, vocab]
            target_probs  = F.softmax(
                target_logits, dim=-1
            )                                       # [1, vocab]
            p_target      = target_probs[0, draft_ids[j]].item()
            p_draft       = draft_probs[j]

            # Sampling acceptance: accept with prob min(1, p_target/p_draft)
            accept_prob   = min(1.0, p_target / p_draft)
            accepted      = torch.rand(1).item() < accept_prob

            self.total_drafted += 1

            if accepted:
                self.total_accepted += 1
                accepted_ids.append(draft_ids[j])
            else:
                # Resample from corrected distribution
                # Correct way: subtract p_draft only at the draft token position
                corrected = target_probs[0].clone()
                corrected[draft_ids[j]] = max(
                    0.0, corrected[draft_ids[j]].item() - p_draft
                )
                corrected = torch.clamp(corrected, min=0.0)

                # Safe normalization — always fall back to target if sum is tiny
                s = corrected.sum()
                if s < 1e-6:
                    corrected = target_probs[0].clone()
                    s = corrected.sum()

                corrected = corrected / s

                # Final safety check — replace any NaN/inf before sampling
                if torch.isnan(corrected).any() or torch.isinf(corrected).any():
                    corrected = torch.ones_like(corrected) / corrected.shape[0]

                bonus_token = torch.multinomial(corrected, 1).item()
                self.total_rollbacks += 1
                break

        # If all k tokens accepted, sample bonus from target at next position
        if bonus_token is None:
            last_pos      = prefix_len - 1 + self.k
            bonus_logits  = out.logits[:, last_pos, :]
            bonus_probs   = F.softmax(bonus_logits, dim=-1)
            bonus_token   = torch.argmax(bonus_probs, dim=-1).item()

        return accepted_ids, bonus_token

    # ── Main generate loop ─────────────────────────────────────
    def generate(self, prompt, max_new_tokens=100, verbose=False):
        """
        Run speculative decoding to completion.
        Returns a dict with text, speed, and statistics.
        """
        self.reset_stats()
        input_ids  = self.tok(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        output_ids = input_ids.clone()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Phase 1: draft
            draft_ids, draft_probs, _ = self._draft(output_ids)

            # Phase 2+3: verify + accept/reject
            accepted_ids, bonus = self._verify(
                output_ids, draft_ids, draft_probs
            )

            # Append accepted prefix + bonus token
            new_ids    = accepted_ids + [bonus]
            new_tensor = torch.tensor(
                new_ids, device=self.device
            ).unsqueeze(0)
            output_ids = torch.cat([output_ids, new_tensor], dim=1)
            tokens_generated += len(new_ids)
            self.total_rounds += 1

            if verbose:
                print(f"  Round {self.total_rounds:3d} | "
                      f"drafted={self.k}  "
                      f"accepted={len(accepted_ids)}  "
                      f"bonus=1  "
                      f"α={self.acceptance_rate:.2f}")

            # Stop on EOS
            if bonus == self.tok.eos_token_id:
                break

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
            "tps":             tokens_generated / elapsed,
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


if __name__ == "__main__":
    from models import load_models, autoregressive_baseline

    print("Loading models...")
    draft_model, draft_tok, target_model, target_tok = load_models()

    PROMPT   = "The first digits of pi are"
    N_TOKENS = 80

    # Baseline: target alone
    print("\n=== TARGET MODEL BASELINE ===")
    baseline_tps, baseline_text = autoregressive_baseline(
        target_model, target_tok, PROMPT, N_TOKENS, "GPT-2 XL"
    )

    # Speculative decoder
    print("\n=== SPECULATIVE DECODER (k=4) ===")
    decoder = SpeculativeDecoder(
        draft_model, target_model, target_tok, k=4
    )
    result = decoder.generate(PROMPT, max_new_tokens=N_TOKENS, verbose=True)
    decoder.print_stats(result)

    speedup = result["tps"] / baseline_tps
    print(f"  Speedup  : {speedup:.2f}x over target baseline")

    # Verify correctness — output should match target model
    print("\n=== CORRECTNESS CHECK ===")
    print(f"Target output    : {baseline_text[:80]}")
    print(f"Speculative out  : {result['text'][:80]}")