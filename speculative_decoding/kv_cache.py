"""
kv_cache.py
Speculative decoder with dual-state KV-cache management.

Key idea:
  - Committed state : verified tokens, immutable, grows monotonically
  - Speculative state: draft tokens, temporary, flushed on rollback

Using the KV-cache means the draft model and target model do NOT
recompute attention over the entire context on every step —
they only process the NEW token(s), reading prior keys/values
from the cache. This is the single biggest speedup lever on 4 GB VRAM.
"""

import torch
import torch.nn.functional as F
import time
from speculative_decoder import SpeculativeDecoder


class KVCacheSpeculativeDecoder(SpeculativeDecoder):
    """
    Extends SpeculativeDecoder with KV-cache support.
    - Draft model uses its own rolling KV-cache during the draft phase.
    - Target model is prefilled once, then uses incremental KV updates.
    - On rollback: KV-cache is trimmed back to the last committed length.
    """

    # ── Trim KV-cache to n_keep tokens ───────────────────────
    @staticmethod
    def _trim_kv(past_key_values, n_keep):
        """
        Slice all KV tensors to keep only the first n_keep positions.
        Shape per layer per key/value: [batch, heads, seq_len, head_dim]
        """
        if past_key_values is None:
            return None
        return tuple(
            tuple(kv[:, :, :n_keep, :] for kv in layer)
            for layer in past_key_values
        )

    # ── Prefill target model on the prompt ───────────────────
    def _prefill_target(self, input_ids):
        """
        Run target model on full prompt once to warm up its KV-cache.
        Returns the KV-cache state after processing the prompt.
        """
        with torch.no_grad():
            out = self.target(
                input_ids,
                use_cache=True,
                return_dict=True,
            )
        return out.past_key_values   # tuple of (key, value) per layer

    # ── Draft with rolling KV-cache ───────────────────────────
    def _draft_with_kv(self, last_token_id):
        """
        Generate k draft tokens using rolling KV-cache for the draft model.
        Much faster than re-running full context each step.
        """
        draft_ids   = []
        draft_probs = []

        # Start from single new token (rest of context is in KV-cache)
        current = torch.tensor(
            [[last_token_id]], device=self.device
        )
        kv = None   # draft model KV accumulates within this call

        with torch.no_grad():
            for _ in range(self.k):
                out    = self.draft(
                    current, past_key_values=kv, use_cache=True
                )
                logits = out.logits[:, -1, :]
                probs  = F.softmax(logits, dim=-1)
                token  = torch.argmax(probs, dim=-1)
                p      = probs[0, token.item()].item()

                draft_ids.append(token.item())
                draft_probs.append(max(p, 1e-9))

                # Next step: feed only the new token
                current = token.unsqueeze(0)
                kv      = out.past_key_values

        return draft_ids, draft_probs

    # ── Verify k draft tokens using target KV-cache ───────────
    def _verify_with_kv(self, target_kv, committed_len,
                        draft_ids, draft_probs):
        """
        Run target model over k draft tokens only.
        The committed context is already in target_kv.

        Returns:
            accepted_ids   : list of accepted token ids
            bonus_token    : int
            updated_kv     : updated target KV-cache (committed + accepted + bonus)
        """
        draft_tensor = torch.tensor(
            draft_ids, device=self.device
        ).unsqueeze(0)                              # [1, k]

        with torch.no_grad():
            out = self.target(
                draft_tensor,
                past_key_values=target_kv,
                use_cache=True,
                return_dict=True,
            )
        verify_logits = out.logits                  # [1, k, vocab]

        accepted_ids = []
        bonus_token  = None
        reject_pos   = None

        for j in range(self.k):
            target_logits = verify_logits[:, j, :]
            target_probs  = F.softmax(target_logits, dim=-1)
            p_target      = target_probs[0, draft_ids[j]].item()
            p_draft       = draft_probs[j]

            accept_prob   = min(1.0, p_target / p_draft)
            accepted      = torch.rand(1).item() < accept_prob

            self.total_drafted += 1

            if accepted:
                self.total_accepted += 1
                accepted_ids.append(draft_ids[j])
            else:
                # Resample bonus from corrected distribution
                # FIXED
                corrected = target_probs[0].clone()
                corrected[draft_ids[j]] = max(
                    0.0, corrected[draft_ids[j]].item() - p_draft
                )
                corrected = torch.clamp(corrected, min=0.0)
                s = corrected.sum()
                if s < 1e-6:
                    corrected = target_probs[0].clone()
                    s = corrected.sum()
                corrected = corrected / s
                if torch.isnan(corrected).any() or torch.isinf(corrected).any():
                    corrected = torch.ones_like(corrected) / corrected.shape[0]
                bonus_token = torch.multinomial(corrected, 1).item()
                reject_pos  = j
                self.total_rollbacks += 1
                break

        if bonus_token is None:
            # All accepted — sample bonus from next position
            last_logits = verify_logits[:, self.k - 1, :]
            last_probs  = F.softmax(last_logits, dim=-1)
            bonus_token = torch.argmax(last_probs, dim=-1).item()

        # Rebuild target KV-cache for committed + accepted tokens
        n_accepted = len(accepted_ids)

        if reject_pos is not None:
            # Trim KV back to committed length, then add bonus
            target_kv = self._trim_kv(
                out.past_key_values, committed_len
            )
        else:
            # All accepted — KV-cache includes all k draft positions
            # Trim to committed + accepted (exclude future speculative)
            target_kv = self._trim_kv(
                out.past_key_values, committed_len + n_accepted
            )

        # Add bonus token to target KV
        bonus_tensor = torch.tensor(
            [[bonus_token]], device=self.device
        )
        with torch.no_grad():
            bonus_out = self.target(
                bonus_tensor,
                past_key_values=target_kv,
                use_cache=True,
                return_dict=True,
            )
        target_kv = bonus_out.past_key_values

        return accepted_ids, bonus_token, target_kv

    # ── Main generate loop with KV-cache ──────────────────────
    def generate(self, prompt, max_new_tokens=100, verbose=False):
        self.reset_stats()

        input_ids = self.tok(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        committed_len = input_ids.shape[1]

        # Prefill target KV-cache with the prompt
        target_kv = self._prefill_target(input_ids)

        # Keep track of last committed token for draft KV seeding
        output_ids = input_ids.clone()
        last_token = input_ids[0, -1].item()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Phase 1: Draft k tokens (KV-accelerated)
            draft_ids, draft_probs = self._draft_with_kv(last_token)

            # Phase 2+3: Verify + accept/reject (KV-accelerated)
            accepted_ids, bonus, target_kv = self._verify_with_kv(
                target_kv, committed_len, draft_ids, draft_probs
            )

            # Commit accepted + bonus
            new_ids    = accepted_ids + [bonus]
            new_tensor = torch.tensor(
                new_ids, device=self.device
            ).unsqueeze(0)
            output_ids = torch.cat([output_ids, new_tensor], dim=1)

            committed_len    += len(new_ids)
            tokens_generated += len(new_ids)
            last_token        = bonus
            self.total_rounds += 1

            if verbose:
                print(f"  Round {self.total_rounds:3d} | "
                      f"accepted={len(accepted_ids)}  bonus=1  "
                      f"α={self.acceptance_rate:.2f}  "
                      f"committed_len={committed_len}")

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

    print("\n=== SPECULATIVE + KV-CACHE (k=4) ===")
    decoder = KVCacheSpeculativeDecoder(
        draft_model, target_model, target_tok, k=4
    )
    result = decoder.generate(PROMPT, max_new_tokens=N_TOKENS, verbose=True)
    decoder.print_stats(result)
    print(f"  Speedup: {result['tps'] / baseline_tps:.2f}x")