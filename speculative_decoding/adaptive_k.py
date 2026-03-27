"""
adaptive_k.py
Adaptive Speculation Depth Controller + Full Speculative Decoder.

Controller rules:
  - Initialize k = max(3, n_tokens // 10)
  - If α (this round) >= alpha_high → k += 1  (slow increase)
  - If α (this round) <  alpha_low  → k -= 2  (fast decrease)
  - Always: k_min <= k <= k_max

The Full Speculative Decoder combines:
  - KV-cache management  (from kv_cache.py)
  - Adaptive k controller (this file)
"""

import torch
import torch.nn.functional as F
import time
from kv_cache import KVCacheSpeculativeDecoder


class AdaptiveKController:
    def __init__(self, k_init, k_min=3, k_max=12,
                 alpha_high=0.72, alpha_low=0.48):
        self.k          = k_init
        self.k_min      = k_min
        self.k_max      = k_max
        self.alpha_high = alpha_high
        self.alpha_low  = alpha_low
        self.history_k  = [k_init]
        self.history_a  = []

    def update(self, round_accepted, round_drafted):
        """
        Called after each round with that round's accept/draft counts.
        Returns the new k value.
        """
        if round_drafted == 0:
            return self.k

        alpha = round_accepted / round_drafted
        self.history_a.append(alpha)

        if alpha >= self.alpha_high:
            self.k = min(self.k + 1, self.k_max)
        elif alpha < self.alpha_low:
            self.k = max(self.k - 2, self.k_min)
        # else: k stays the same

        self.history_k.append(self.k)
        return self.k

    def summary(self):
        if not self.history_a:
            return
        avg_a = sum(self.history_a) / len(self.history_a)
        avg_k = sum(self.history_k) / len(self.history_k)
        print(f"\n  Adaptive K summary:")
        print(f"    Final k   : {self.k}")
        print(f"    Average k : {avg_k:.1f}")
        print(f"    Average α : {avg_a:.2%}")
        print(f"    k history : {self.history_k[:20]}{'...' if len(self.history_k)>20 else ''}")


class FullSpeculativeDecoder(KVCacheSpeculativeDecoder):
    """
    Full Speculative Decoder:
      KV-cache + Adaptive speculation depth.
    This is the final, best-performing configuration.
    """

    def __init__(self, draft_model, target_model, tokenizer,
                 k_init=4, k_min=3, k_max=12,
                 alpha_high=0.72, alpha_low=0.48):
        super().__init__(draft_model, target_model, tokenizer, k=k_init)
        self.controller = AdaptiveKController(
            k_init, k_min, k_max, alpha_high, alpha_low
        )

    def generate(self, prompt, max_new_tokens=100, verbose=False):
        self.reset_stats()
        self.controller = AdaptiveKController(
            self.k, self.controller.k_min, self.controller.k_max,
            self.controller.alpha_high, self.controller.alpha_low
        )

        input_ids     = self.tok(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        committed_len = input_ids.shape[1]
        target_kv     = self._prefill_target(input_ids)
        output_ids    = input_ids.clone()
        last_token    = input_ids[0, -1].item()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_generated  = 0
        round_accepted_   = 0
        round_drafted_    = 0

        while tokens_generated < max_new_tokens:
            # Use current k from adaptive controller
            self.k = self.controller.k

            # Phase 1: Draft
            draft_ids, draft_probs = self._draft_with_kv(last_token)

            # Phase 2+3: Verify
            prev_accepted = self.total_accepted
            prev_drafted  = self.total_drafted

            accepted_ids, bonus, target_kv = self._verify_with_kv(
                target_kv, committed_len, draft_ids, draft_probs
            )

            # Track this round's stats for the controller
            round_drafted_  = self.total_drafted  - prev_drafted
            round_accepted_ = self.total_accepted - prev_accepted

            # Update adaptive k
            new_k = self.controller.update(round_accepted_, round_drafted_)

            # Commit
            new_ids    = accepted_ids + [bonus]
            new_tensor = torch.tensor(
                new_ids, device=self.device
            ).unsqueeze(0)
            output_ids    = torch.cat([output_ids, new_tensor], dim=1)
            committed_len += len(new_ids)
            tokens_generated += len(new_ids)
            last_token        = bonus
            self.total_rounds += 1

            if verbose:
                print(f"  Round {self.total_rounds:3d} | "
                      f"k={self.k}->{new_k}  "
                      f"accepted={len(accepted_ids)}  "
                      f"α={self.acceptance_rate:.2f}")

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
            "k_final":         self.controller.k,
            "k_avg":           sum(self.controller.history_k) / max(len(self.controller.history_k), 1),
        }


if __name__ == "__main__":
    from models import load_models, autoregressive_baseline

    print("Loading models...")
    draft_model, draft_tok, target_model, target_tok = load_models()

    PROMPT   = "The first digits of pi are"
    N_TOKENS = 80

    print("\n=== TARGET BASELINE ===")
    baseline_tps, _ = autoregressive_baseline(
        target_model, target_tok, PROMPT, N_TOKENS, "GPT-2 XL"
    )

    print("\n=== FULL SPECULATIVE DECODER (KV + Adaptive K) ===")
    decoder = FullSpeculativeDecoder(
        draft_model, target_model, target_tok, k_init=4
    )
    result = decoder.generate(PROMPT, max_new_tokens=N_TOKENS, verbose=True)
    decoder.print_stats(result)
    decoder.controller.summary()
    print(f"\n  Speedup: {result['tps'] / baseline_tps:.2f}x")