import numpy as np
import time
import random
import heapq
import matplotlib.pyplot as plt


class BanditCampaignOptimizer:
    """
    Multi-armed bandit approach for campaign autotargeting.

    Each campaign is treated as an arm. Users and items are assigned to campaigns
    using bandit strategies that balance exploration (trying underexplored campaign
    compositions) with exploitation (assigning to the highest-affinity campaign).

    The core loop alternates between user and item assignment phases (like the OG
    algorithm), but replaces greedy allocation with bandit-augmented scoring:

        bandit_score[entity, campaign] = affinity_score + exploration_bonus

    Three strategies are supported:
      - 'thompson':       Thompson Sampling - Gaussian noise scaled by uncertainty
      - 'ucb':            Upper Confidence Bound (UCB1)
      - 'epsilon_greedy': Epsilon-greedy with exponential decay
    """

    def __init__(self, user_embeds, item_embeds, item_biases, ad_configs, items_per_ad=5):
        """
        Same interface as AutoTargetingEngine.

        :param user_embeds:  (n_users, d) array of user embeddings
        :param item_embeds:  (n_items, d) array of item embeddings
        :param item_biases:  (n_items,) array of item biases (unused, kept for API compat)
        :param ad_configs:   list of dicts with 'id', 'min_users', 'max_users', etc.
        :param items_per_ad: target number of items per campaign
        """
        self.users = user_embeds
        self.items = item_embeds
        self.ad_configs = ad_configs
        self.items_per_ad = items_per_ad

        self.n_users = len(user_embeds)
        self.n_items = len(item_embeds)
        self.n_ads = len(ad_configs)

        if self.n_ads > 15:
            raise ValueError("Max 15 ads allowed")

        self.TOMBSTONE_IDX = self.n_ads

        # Affinity matrix: exp(user_embeds @ item_embeds.T)
        self.affinity_matrix = np.exp(self.users @ self.items.T)

        # Assignment state
        self.user_assignments = np.full(self.n_users, self.TOMBSTONE_IDX, dtype=int)
        self.item_assignments = np.full(self.n_items, self.TOMBSTONE_IDX, dtype=int)

        # Per-campaign bandit statistics (separate for user and item phases)
        self.user_stats = _make_stats(self.n_ads)
        self.item_stats = _make_stats(self.n_ads)

        self.score_history = []
        self.regret_history = []   # two-sided regret decomposition per round

        print(f"Bandit Engine Init: {self.n_ads} campaigns, "
              f"{self.n_users} users, {self.n_items} items, "
              f"{self.items_per_ad} items/campaign")

    # ------------------------------------------------------------------
    # Main solver
    # ------------------------------------------------------------------

    def solve(self, n_rounds=50, strategy='thompson', epsilon=0.1,
              ucb_c=2.0, epsilon_decay=0.95, stats_window=0,
              verbose=True):
        """
        Run the bandit optimization.

        Each round:
          1. Fix items  -> assign users via bandit
          2. Fix users  -> assign items via bandit
          3. Evaluate global score and track best

        Args:
            n_rounds:             max optimization rounds
            strategy:             'thompson' | 'ucb' | 'epsilon_greedy'
            epsilon:              exploration rate (epsilon-greedy)
            ucb_c:                exploration constant (UCB)
            epsilon_decay:        per-round decay for epsilon
            stats_window:         rolling window size for bandit stats (0 = no decay).
                                  each round, stats are decayed by (1 - 1/window) so
                                  observations older than ~window rounds carry negligible
                                  weight. handles non-stationarity without losing all history.
            verbose:              print progress

        Returns:
            (best_user_assignments, best_item_assignments, best_score)
        """
        self._random_item_scatter()
        self._record_history(0, "Start")

        best_score = -np.inf
        best_users = None
        best_items = None
        no_improvement = 0
        current_epsilon = epsilon
        PATIENCE = 10

        if verbose:
            print(f"--- Bandit Optimization (strategy={strategy}, "
                  f"rounds={n_rounds}) ---")

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()

            # Rolling window: decay stats so older observations lose weight
            if stats_window > 0:
                decay = 1.0 - 1.0 / stats_window
                _decay_stats(self.user_stats, decay)
                _decay_stats(self.item_stats, decay)

            # Score before any reassignment this round (end of previous round)
            score_before_users = self._calculate_global_score()

            # Phase 1: assign users (items fixed)
            user_scores = self._aggregate_scores(target='users')
            self._bandit_assign(
                scores=user_scores,
                assignments=self.user_assignments,
                constraints=self.ad_configs,
                capacity_key='max_users',
                stats=self.user_stats,
                strategy=strategy,
                round_num=rnd,
                epsilon=current_epsilon,
                ucb_c=ucb_c,
            )

            # Score after user phase only (items still fixed)
            score_after_users = self._calculate_global_score()

            # Phase 2: assign items (users fixed)
            item_scores = self._aggregate_scores(target='items')
            item_constraints = [
                {'max_items': self.items_per_ad, 'id': k}
                for k in range(self.n_ads)
            ]
            self._bandit_assign(
                scores=item_scores,
                assignments=self.item_assignments,
                constraints=item_constraints,
                capacity_key='max_items',
                stats=self.item_stats,
                strategy=strategy,
                round_num=rnd,
                epsilon=current_epsilon,
                ucb_c=ucb_c,
            )

            # Evaluate
            total_score = self._calculate_global_score()
            self._record_history(rnd, f"Round {rnd}")

            # Two-sided regret decomposition:
            #   user_delta: score gain attributable to user phase exploration
            #   item_delta: score gain attributable to item phase exploration
            #   instant_regret: gap from best known (proxy for OPT)
            #   cumul_regret: R(T) = sum_t [OPT - score_t], bounded by R_user(T) + R_item(T)
            user_delta = score_after_users - score_before_users
            item_delta = total_score - score_after_users
            prior_cumul = self.regret_history[-1]['cumul_regret'] if self.regret_history else 0.0
            self.regret_history.append({
                'round':          rnd,
                'score_before':   score_before_users,
                'score_after_users': score_after_users,
                'score_after_items': total_score,
                'user_delta':     user_delta,
                'item_delta':     item_delta,
                'instant_regret': max(best_score, total_score) - total_score,
                'cumul_regret':   prior_cumul + max(best_score - total_score, 0.0),
            })

            improvement = total_score - best_score
            if verbose and (rnd <= 5 or rnd % 5 == 0 or improvement > 1e-4):
                print(f"  Round {rnd:>3}: Score={total_score:.8e} "
                      f"(Delta={improvement:+.5e}) [{time.time()-t0:.2f}s]"
                      + (f"  eps={current_epsilon:.4f}"
                         if strategy == 'epsilon_greedy' else ""))

            if total_score > best_score + 1e-4:
                best_score = total_score
                best_users = self.user_assignments.copy()
                best_items = self.item_assignments.copy()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= PATIENCE:
                    if verbose:
                        print(f"  Converged after {rnd} rounds "
                              f"(no improvement for {PATIENCE}).")
                    break

            if strategy == 'epsilon_greedy':
                current_epsilon *= epsilon_decay

        # Restore best configuration
        self.user_assignments = best_users
        self.item_assignments = best_items

        # Retroactively recompute cumulative regret using final best_score as
        # OPT proxy.  During the loop, best_score was updated *after* recording
        # each entry, so early rounds were measured against a moving (too-low)
        # ceiling.  Using the final ceiling gives a fair, comparable R(T) across
        # all strategies: UCB no longer shows 0 just because it never regresses.
        opt_proxy = best_score
        cumul = 0.0
        for entry in self.regret_history:
            gap = max(opt_proxy - entry['score_after_items'], 0.0)
            entry['instant_regret'] = gap
            cumul += gap
            entry['cumul_regret'] = cumul

        return best_users, best_items, best_score

    # ------------------------------------------------------------------
    # Bandit assignment logic
    # ------------------------------------------------------------------

    def _bandit_assign(self, scores, assignments, constraints, capacity_key,
                       stats, strategy, round_num, epsilon, ucb_c):
        """
        Assign entities to campaigns using bandit-augmented scores.

        1. Compute bandit scores = raw affinity + exploration bonus
        2. Heap-based greedy assignment respecting capacity constraints
        3. Update bandit statistics with observed rewards
        """
        n_entities = len(assignments)

        # Bandit-augmented scores
        bandit_scores = self._compute_bandit_scores(
            scores, stats, strategy, round_num, epsilon, ucb_c
        )

        # Reset assignments
        assignments[:] = self.TOMBSTONE_IDX
        counts = np.zeros(self.n_ads, dtype=int)

        # Build priority queue: each entity starts with its best campaign
        # Heap entries: (neg_bandit_score, entity_idx, campaign_idx)
        heap = []
        for e in range(n_entities):
            best_k = int(np.argmax(bandit_scores[e]))
            heapq.heappush(heap, (-bandit_scores[e, best_k], e, best_k))

        assigned = np.zeros(n_entities, dtype=bool)

        while heap:
            neg_score, e_idx, k_idx = heapq.heappop(heap)

            if assigned[e_idx]:
                continue

            max_cap = constraints[k_idx].get(capacity_key, float('inf'))

            if counts[k_idx] < max_cap:
                # Assign entity to campaign
                assignments[e_idx] = k_idx
                counts[k_idx] += 1
                assigned[e_idx] = True

                # Update bandit statistics with actual reward
                reward = scores[e_idx, k_idx]
                stats['counts'][k_idx] += 1
                stats['sum_rewards'][k_idx] += reward
                stats['sum_sq_rewards'][k_idx] += reward ** 2
            else:
                # Campaign full - try next best
                bandit_scores[e_idx, k_idx] = -np.inf
                remaining = bandit_scores[e_idx]
                next_k = int(np.argmax(remaining))
                if remaining[next_k] > -np.inf:
                    heapq.heappush(heap, (-remaining[next_k], e_idx, next_k))

    def _compute_bandit_scores(self, scores, stats, strategy,
                               round_num, epsilon, ucb_c):
        """Dispatch to the appropriate bandit scoring strategy."""
        if strategy == 'thompson':
            return self._thompson_scores(scores, stats)
        elif strategy == 'ucb':
            return self._ucb_scores(scores, stats, round_num, ucb_c)
        elif strategy == 'epsilon_greedy':
            return self._epsilon_greedy_scores(scores, epsilon)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Use 'thompson', 'ucb', or 'epsilon_greedy'.")

    def _thompson_scores(self, scores, stats):
        """
        Thompson Sampling: sample from a Gaussian posterior per campaign.

        For campaign k with n_k observations:
            sigma_k = 1 / sqrt(n_k + 1)
            noise ~ N(0, sigma_k * score_scale_k)
            bandit_score[e, k] = score[e, k] + noise[e]

        The noise is proportional to score magnitude so exploration scales
        appropriately regardless of the affinity range.
        """
        augmented = scores.copy()
        n_entities = scores.shape[0]

        for k in range(self.n_ads):
            n_k = stats['counts'][k]
            sigma = 1.0 / np.sqrt(n_k + 1)
            score_scale = np.mean(np.abs(scores[:, k])) + 1e-8
            noise = np.random.normal(0, sigma * score_scale, size=n_entities)
            augmented[:, k] += noise

        return augmented

    def _ucb_scores(self, scores, stats, round_num, ucb_c):
        """
        UCB1: add upper confidence bound as exploration bonus.

            bonus_k = c * sqrt(ln(t) / n_k)
            bandit_score[e, k] = score[e, k] + bonus_k * score_scale_k
        """
        augmented = scores.copy()

        for k in range(self.n_ads):
            n_k = max(stats['counts'][k], 1)
            bonus = ucb_c * np.sqrt(np.log(round_num + 1) / n_k)
            score_scale = np.mean(np.abs(scores[:, k])) + 1e-8
            augmented[:, k] += bonus * score_scale

        return augmented

    def _epsilon_greedy_scores(self, scores, epsilon):
        """
        Epsilon-greedy: with probability epsilon, replace scores with
        uniform random values to force exploration.
        """
        augmented = scores.copy()
        n_entities = scores.shape[0]

        explore_mask = np.random.random(n_entities) < epsilon
        n_explore = int(np.sum(explore_mask))
        if n_explore > 0:
            lo = scores[explore_mask].min()
            hi = scores[explore_mask].max()
            augmented[explore_mask] = np.random.uniform(
                lo, hi, size=(n_explore, self.n_ads)
            )

        return augmented

    # ------------------------------------------------------------------
    # Score aggregation (same as OG)
    # ------------------------------------------------------------------

    def _aggregate_scores(self, target):
        """
        Aggregate affinity scores by current opposite-side assignments.

        For users:  score[u, k] = sum(affinity[u, i]  for i assigned to k)
        For items:  score[i, k] = sum(affinity[u, i]  for u assigned to k)
        """
        if target == 'users':
            dist = np.zeros((self.n_items, self.n_ads), dtype=np.float32)
            for idx, ad in enumerate(self.item_assignments):
                if ad != self.TOMBSTONE_IDX:
                    dist[idx, ad] = 1.0
            return self.affinity_matrix @ dist
        else:
            dist = np.zeros((self.n_users, self.n_ads), dtype=np.float32)
            for idx, ad in enumerate(self.user_assignments):
                if ad != self.TOMBSTONE_IDX:
                    dist[idx, ad] = 1.0
            return self.affinity_matrix.T @ dist

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _random_item_scatter(self):
        """Randomly assign items to campaigns up to items_per_ad."""
        counts = [0] * self.n_ads
        self.item_assignments[:] = self.TOMBSTONE_IDX
        indices = list(range(self.n_items))
        random.shuffle(indices)
        for i in indices:
            ad = random.randint(0, self.n_ads - 1)
            if counts[ad] < self.items_per_ad:
                self.item_assignments[i] = ad
                counts[ad] += 1
            else:
                for other_ad in range(self.n_ads):
                    if counts[other_ad] < self.items_per_ad:
                        self.item_assignments[i] = other_ad
                        counts[other_ad] += 1
                        break

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _calculate_global_score(self):
        """
        Total quality = sum of bicluster means (same as OG / biclustering).

        For each campaign, extract the (users_in_k x items_in_k) submatrix
        of the affinity matrix and take its mean.
        """
        total = 0.0
        for k in range(self.n_ads):
            u_idx = np.where(self.user_assignments == k)[0]
            i_idx = np.where(self.item_assignments == k)[0]
            if len(u_idx) > 0 and len(i_idx) > 0:
                total += np.mean(self.affinity_matrix[np.ix_(u_idx, i_idx)])
        return total

    def _record_history(self, step, label):
        self.score_history.append({
            'step': step,
            'score': self._calculate_global_score(),
            'label': label,
        })

    # ------------------------------------------------------------------
    # Reporting (same format as OG)
    # ------------------------------------------------------------------

    def plot_two_sided_regret(self):
        """
        Visualise the two-sided regret decomposition.

        Two-sided regret bound:
            R(T) = sum_t [OPT - score_t]
                 <= R_user(T) + R_item(T)

        where
            R_user(T) = sum_t max(0, user_delta_t lost to exploration)
            R_item(T) = sum_t max(0, item_delta_t lost to exploration)

        Plotting the per-round user_delta and item_delta shows how much
        of each round's score movement came from each side of the assignment,
        and the cumulative regret curve shows convergence to OPT.
        """
        if not self.regret_history:
            print("No regret history - run solve() first.")
            return

        rounds      = [r['round']        for r in self.regret_history]
        user_deltas = [r['user_delta']   for r in self.regret_history]
        item_deltas = [r['item_delta']   for r in self.regret_history]
        cumul       = [r['cumul_regret'] for r in self.regret_history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-round score contributions by phase
        axes[0].bar(rounds, user_deltas, label='User phase delta',
                    color='#2196F3', alpha=0.7)
        axes[0].bar(rounds, item_deltas, bottom=user_deltas,
                    label='Item phase delta', color='#FF9800', alpha=0.7)
        axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[0].set_xlabel('Round', fontsize=12)
        axes[0].set_ylabel('Score Delta', fontsize=12)
        axes[0].set_title('Two-Sided Score Contributions per Round')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Right: cumulative regret R(T) = sum_t [OPT - score_t]
        axes[1].plot(rounds, cumul, color='#E53935', linewidth=2, marker='o',
                     markersize=3)
        axes[1].set_xlabel('Round T', fontsize=12)
        axes[1].set_ylabel('Cumulative Regret R(T)', fontsize=12)
        axes[1].set_title('Cumulative Two-Sided Regret')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        steps = [h['step'] for h in self.score_history]
        scores = [h['score'] for h in self.score_history]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, scores, marker='o', linestyle='-',
                 color='#007acc', linewidth=2)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Affinity Score', fontsize=12)
        plt.title('Bandit Campaign Optimizer - Convergence')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def print_report(self):
        """
        Print per-campaign quality, utility, lift and overall Gini coefficient.
        Identical output format to AutoTargetingEngine.print_report().
        """
        total_mean_quality = 0.0
        total_global_utility = 0.0
        valid_ads = 0
        global_mean = np.mean(self.affinity_matrix)
        per_user_utils = np.zeros(self.n_users)

        print(f"\n{'Ad ID':<6} | {'Dim (Users x Items)':<22} | "
              f"{'Quality (Mean)':<18} | {'Utility (Sum)':<18} | {'Lift':<10}")
        print("-" * 85)

        for ad_idx in range(self.n_ads):
            u_idx = np.where(self.user_assignments == ad_idx)[0]
            i_idx = np.where(self.item_assignments == ad_idx)[0]

            if len(u_idx) > 0 and len(i_idx) > 0:
                valid_ads += 1
                sub = self.affinity_matrix[np.ix_(u_idx, i_idx)]
                mean_s = np.mean(sub)
                sum_s = np.sum(sub)
                lift = mean_s / global_mean if global_mean > 0 else 0.0
                per_user_utils[u_idx] = np.mean(sub, axis=1)

                total_mean_quality += mean_s
                total_global_utility += sum_s

                dims = f"{len(u_idx)} x {len(i_idx)}"
                print(f"{ad_idx:<6} | {dims:<22} | {mean_s:<18.4f} | "
                      f"{sum_s:<18.4f} | {lift:<10.2f}x")

        print("-" * 85)
        avg_mean = total_mean_quality / valid_ads if valid_ads > 0 else 0.0
        avg_lift = avg_mean / global_mean if global_mean > 0 else 0.0

        assigned_mask = self.user_assignments != self.TOMBSTONE_IDX
        gini = gini_coefficient(per_user_utils[assigned_mask])

        print(f"AVG Quality (Mean of Means): {total_mean_quality:.4f}")
        print(f"TOT Utility (Global Sum):    {total_global_utility:.4f}")
        print(f"AVG Lift:                    {avg_lift:.2f}x")
        print(f"Gini Coefficient:            {gini:.4f}")

        # Two-sided regret summary
        # R_item(T) = sum_t max(OPT - score_after_items_t, 0)  = cumul_regret
        # R_user(T) = sum_t max(OPT - score_after_users_t, 0)  >= R_item(T)
        #
        # Relationship: R(T) = R_item(T) <= R_user(T)
        # Gap R_user(T) - R_item(T) = total score recovered by the item phase.
        # A large gap means items do heavy lifting; a small gap means item
        # exploration cost is low relative to user exploration cost.
        if self.regret_history:
            opt_proxy    = max(r['score_after_items'] for r in self.regret_history)
            r_user       = sum(max(opt_proxy - r['score_after_users'], 0.0) for r in self.regret_history)
            r_item       = self.regret_history[-1]['cumul_regret']   # = R(T)
            item_recovery = r_user - r_item
            print(f"\n--- Two-Sided Regret Decomposition ---")
            print(f"R(T)  cumulative regret:     {r_item:.4f}")
            print(f"R_user(T) [after user phase]:{r_user:.4f}")
            print(f"Item-phase recovery:         {item_recovery:.4f}  (R_user - R_item)")

        return avg_mean, total_global_utility


# ======================================================================
# Helpers
# ======================================================================

def _make_stats(n_ads):
    """Create a fresh bandit statistics dict."""
    return {
        'counts': np.zeros(n_ads, dtype=np.float64),
        'sum_rewards': np.zeros(n_ads, dtype=np.float64),
        'sum_sq_rewards': np.zeros(n_ads, dtype=np.float64),
    }


def _decay_stats(stats, decay):
    """Apply exponential decay to bandit statistics (rolling window)."""
    stats['counts'] *= decay
    stats['sum_rewards'] *= decay
    stats['sum_sq_rewards'] *= decay


def gini_coefficient(values):
    """Gini coefficient. 0 = perfect equality, 1 = max inequality."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
