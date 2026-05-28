# LLM Council Session — Swarm method, reward model, LSTM PPO decision
**Date:** 2026-05-28 18:06:54
**Repo:** isaac_drone_racer
**Branch:** master

---

## Original question
> "council the swarm method and its reward model plus PPO model should switch to lstm ppo or not"

---

## Framed question

Project: Isaac Drone Racer 2 (Isaac Sim 5.1 + Isaac Lab 2.3.2 + skrl). Single-drone PPO already trains successfully. User added Swarm tasks: N=4 drones in shared env, single batched PPO over concatenated per-drone state (NOT a true shared/permutation-invariant policy), CTBR action interface, asymmetric actor-critic.

Current reward stack (paper-adapted from Geles et al 2024 "Superhuman Safe and Agile Racing"):
- progress (+20, signed dist delta), gate_passed (+400 sparse, no miss penalty), lookat (+0.1, heading exp), ang_vel L1 (-0.01), rank (+2, paper Eq.4), proximity (curriculum 0→-1 at step 500k, paper Eq.3 vel-weighted exp), terminating (-25 per drone). Drone-drone collision termination REMOVED — soft penalty only. 10% of d2d hits forgiven for recovery learning.

Latest training run (100 PPO updates, 4 drones): total reward flipped positive (-2.14 → +3.46), policy stddev collapsing (1.01 → 0.51, good sign), value loss 0.40 → 0.12. BUT: gate_passed ≈ 0.02 per episode (rare passes), episode length 60 steps (≈0.6s — drones crash fast), lookat ≈ 0 (not orientating to gates).

Paper uses: recurrent PPO (1 LSTM + 2 FC) + Perceiver attention opponent encoder + league self-play (75% past checkpoints, 25% fixed diverse pool) + downwash particle model.

Hardware: RTX 4090 primary, RTX 3060 6/12 GB lite preset. Iteration cost: single update ≈ 5–10s wall on 4090 with 256 envs.

Decisions on the table:
1. Is the current swarm architecture (concatenated state MLP PPO) viable, or must it switch to LSTM PPO + attention opponent encoder to get past the gate-pass barrier?
2. Is the paper-adapted reward stack correct, or are the weights miscalibrated for current training scale (60-step episodes, 256 envs)?
3. Should LSTM upgrade be done now (before fixing reward signal) or only after single-drone variant clears gates reliably?

---

## Anonymization map (revealed)
- **Response A** = First Principles Thinker
- **Response B** = Executor
- **Response C** = Contrarian
- **Response D** = Expansionist
- **Response E** = Outsider

---

## Advisor responses

### Contrarian (was Response C)
Your training isn't broken because you lack an LSTM. It's broken because **your drones are dying in 0.6 seconds and you're debating architecture.**

Look at the actual signal: episode length 60 steps, gate_passed 0.02, lookat ≈ 0. The policy never gets to a gate. It's not failing to *remember* — it's failing to *survive*. An LSTM gives memory to a policy that has nothing worth remembering yet. You'll burn a week on recurrent PPO and the gate-pass rate will still be 0.02 because the underlying problem is upstream of memory.

The reward stack is the obvious suspect and you're under-interrogating it. Progress (+20) signed-delta over a 60-step episode is paying the policy to *exist near gates*, not pass them. Gate_passed (+400 sparse) is unreachable noise at this episode length. Terminating (-25/drone) plus soft d2d collisions plus 90% non-forgiven hits = you removed the hard collision term and replaced it with a death-by-thousand-cuts that still kills episodes. The reward is positive because progress + rank + lookat baseline outweighs termination — **the policy learned to farm shaping rewards before dying**, which is exactly what stddev collapse on a 0.6s episode means. That's not convergence. That's local optimum lock-in.

Also: you're benchmarking a 4-drone swarm before the single-drone variant clears gates reliably. Why? You buried the control variable. If single-drone PPO on this same reward stack doesn't pass gates either, the swarm isn't the problem and neither is the architecture.

**Order of operations:** single-drone first, fix episode length, then decide if LSTM is worth weeks. Don't rewrite to find out reward was the bug.

### First Principles Thinker (was Response A)
You're asking the wrong question entirely.

The question isn't "MLP vs LSTM" or "reward weights." The question is: **why are your drones dying in 0.6 seconds, and does any of this RL machinery matter until they don't?**

60-step episodes mean your agents are not solving a racing problem. They are solving a "fall down slightly slower" problem. Gate-passing reward at +400 is irrelevant if the policy never sees a gate. The progress signal is dominated by termination noise. LSTM won't help — there's no temporal structure to model in 60 steps. Attention over opponents won't help — opponents aren't the reason you're crashing. Reward reweighting won't help — the credit assignment horizon is shorter than the skill horizon.

Strip it down. What are you actually trying to learn? *Stable forward flight toward a target.* That's the atomic skill. You don't have it.

Before any swarm work, any LSTM, any Perceiver: **does your single-drone NoCam policy reliably clear gate 1 from a stationary hover?** If yes, gate 2? If no, your CTBR action mapping, motor allocation, or termination conditions are broken — not your reward stack, not your architecture.

The paper you're copying assumes a working single-agent flight controller as a prior. Geles et al. didn't start from scratch — they had agile single-drone policies first. You're trying to learn swarm racing and basic flight simultaneously with shared gradients. That's the actual bug.

Concrete reframe: kill the swarm runs. Prove single-drone clears 3 gates consecutively in >300-step episodes. Then add a second drone. Then worry about LSTM. The architectural debate is premature optimization on a foundation that hasn't been verified to hold weight.

### Expansionist (was Response D)
Stop framing this as "fix gates first or upgrade architecture first." That's small thinking. You're sitting on something way more valuable than a gate-passing policy.

**The 60-step crash episodes are gold, not a bug.** You have a swarm env that generates dense collision/recovery data faster than anyone publishing in this space. Geles et al. needed a full league + Perceiver to get there. You can leapfrog by treating those crashes as the primary training signal — curriculum bootstraps from "survive 2s near 3 opponents" before you ever care about gates. Reframe the task and 60-step episodes become a feature.

**Bigger play: ship the LSTM + attention encoder NOW, not later.** Here's what everyone's missing — once you have a recurrent permutation-invariant policy that works on N=4, the same checkpoint trivially scales to N=8, N=16, mixed-skill leagues, even cross-embodiment (quad vs. fixed-wing opponents). The single-drone variant is a dead-end artifact. You're not upgrading architecture to fix gates; you're unlocking an entire research axis (emergent blocking, drafting, downwash exploitation) that an MLP literally cannot represent.

**Hidden upside #1:** Your 5–10s/update on a 4090 means you can run league self-play *overnight* at a scale most labs can't. Geles used a TPU cluster. You have an iteration advantage they didn't.

**Hidden upside #2:** Asymmetric AC + privileged critic + LSTM actor = you're 80% of the way to a sim-to-real distillation pipeline. The cam-based student is a paper of its own.

**Hidden upside #3:** Downwash model + soft d2d penalty = first open swarm racing benchmark. Release it.

Do the LSTM upgrade. Stop optimizing the wrong objective.

### Outsider (was Response E)
Step back. You have a 60-step episode length. That's 0.6 seconds. The drones aren't failing to learn gates — they're not surviving long enough to *see* gates. Everything else is noise on top of that one number.

Before touching architecture, answer this: what's killing them at 60 steps? Is it ground impact, drone-drone collision, going out of bounds, or the -25 terminating penalty firing on something benign? You haven't said. If you don't know, you're optimizing blind.

The reward stack tells a story you may not be hearing. Progress went positive, value loss dropped, stddev collapsed — the policy is confidently learning *something*. But gate_passed is ~0.02 and lookat is ~0. Translation: it found a local optimum that farms progress reward without ever needing to orient toward a gate. That's a reward-shaping bug, not an architecture bug. Progress at +20 with gate_passed at +400 sparse means the dense signal dominates and the sparse one is unreachable in 60 steps. The policy is doing exactly what you incentivized.

On the LSTM question: you're asking whether to rewrite the engine because the car won't start. The paper's recurrent + attention + league stack is what you need to beat *other policies*, not what you need to pass a gate. Single-drone PPO already passes gates with an MLP. The swarm version isn't failing at the swarm part — it's failing at the solo part, in a swarm setting.

Order of operations: instrument termination causes first. Then rebalance progress vs. gate_passed (or add a per-step survival bonus) until episodes reach 200+ steps and gate_passed climbs. Only then ask whether LSTM helps. Don't rewrite architecture to fix a thermostat.

### Executor (was Response B)
Stop. You're debating LSTM vs MLP while your drones die in 0.6 seconds. That is not a policy architecture problem. That is a "you cannot fly" problem.

Monday morning, in order:

**1. Diagnose the 60-step death.** Run play with `--log 5`, watch ONE drone. Is it crashing into the ground? The next drone? Spinning out? You don't know yet, and no LSTM fixes a problem you haven't named. 30 minutes of work.

**2. Train the single-drone variant to gate-pass competence FIRST.** You said it "trains successfully" — does it actually clear gates reliably? If a 1-drone MLP can't pass gates with your current reward stack, a 4-drone LSTM won't either. This is the cheapest possible ablation. Run it tonight on the 4090.

**3. Reward weights are almost certainly the problem, not the architecture.** gate_passed=+400 sparse with 0.02/episode hit rate means the policy sees that signal ~once per 50 episodes. Meanwhile progress=+20 dense fires every step. Your dense terms are drowning the sparse one. Either crank gate_passed to +2000, or shape progress to decay near gate so the agent has to actually pass to keep earning. Also: terminating=-25 over 60 steps = -0.42/step baseline cost the policy is desperately trying to escape by... crashing faster. Check that.

**4. Do NOT touch LSTM/Perceiver/league self-play yet.** That's 2 weeks of plumbing. You haven't earned the right to those tools until single-drone clears gates and 4-drone-MLP plateaus for a clear reason.

Order: diagnose crashes → fix single-drone → rebalance rewards → THEN consider recurrence. Anything else is procrastination dressed as engineering.

---

## Peer reviews (anonymized — reviewers saw A/B/C/D/E)

### Reviewer 1
**1. Strongest: B.** It's the only one that converts the shared diagnosis into a concrete, ordered Monday-morning plan: run `--log 5` on one drone, ablate single-drone first, then attack the specific reward-scale math (gate_passed once per 50 episodes vs dense progress every step; -25/60 = -0.42/step coercing faster crashes). C and E reach the same conclusion but stay diagnostic. A is correct but abstract. B is actionable.

**2. Biggest blind spot: D.** It reframes 60-step crashes as "gold" and pushes LSTM + attention + league self-play before the policy can fly 2 seconds. This is exactly the premature-optimization trap A, B, C, E correctly call out. "Sim-to-real distillation pipeline" and "open swarm benchmark" on a policy that dies in 0.6s is fantasy. It's also wrong that the single-drone variant is a "dead-end artifact" — it's the control experiment that isolates whether the bug is swarm-specific.

**3. What all five missed:** None questioned whether the 60-step termination is actually a *learned* crash vs. a spawn-time artifact. Recent commits (`fix(asset): zero prop mass`, `bump contact force=80`) show ongoing contact-sensor phantoms. If `illegal_contact` fires at reset from residual prop contacts, no reward tuning or architecture change fixes it. First check: log which termination term fires at step 60, and whether it's non-zero at step 0–5.

### Reviewer 2
**1. Strongest: B.** Only one with a concrete Monday-morning sequence: diagnose with `--log 5`, confirm single-drone gate competence, then rebalance. Names the specific math (terminating=-25 over 60 steps = -0.42/step incentivizing faster crashes). A and C reach the same conclusion but stop at "strip it down"; B gives executable steps. E is close but vaguer on numbers.

**2. Biggest blind spot: D.** Frames 0.6s crash episodes as "gold" and pushes LSTM+attention+league self-play now. This is exactly the premature-optimization trap A/B/C/E correctly diagnose. You cannot distill a cam-student from a teacher that doesn't fly. "Leapfrog Geles" without a working single-agent controller is fantasy. The hidden-upsides framing rationalizes building on a broken foundation.

**3. What ALL five missed:** Nobody questioned the **CTBR action mapping and motor allocation itself**. A 60-step death across 4 drones with stddev collapsing suggests a possible **spawn-state or allocation bug** — drones may be entering unrecoverable states at reset (prop spin-up transient, IMU warmup, gate-relative spawn pose). Before reward or architecture debates: log the first 10 steps of one episode and verify the drone is physically flyable from t=0.

### Reviewer 3
**1. Strongest: E.** E names the actual diagnostic signature: progress↑ + value_loss↓ + stddev↓ + gate_passed≈0 + lookat≈0 = confident local optimum farming dense progress without orienting. That's a falsifiable hypothesis tied to the numbers given, not just vibes. B is close and more actionable (concrete weights, -0.42/step death-cost math), but E's diagnosis is sharper. A and C reach the same conclusion (single-drone first) with less mechanism. D is reckless given gate_passed=0.02.

**2. Biggest blind spot: D.** Shipping LSTM+attention+league self-play on a policy that can't pass a gate in 60 steps is architecture-as-cargo-cult. "60-step crashes are GOLD" inverts the signal — they're noise from a broken flight prior, not collision/recovery data. League self-play over policies that crash in 0.6s produces a league of crashers. D also ignores that lookat≈0 means no gate orientation exists to make permutation-invariant over.

**3. What all five missed:** Nobody questioned the **10% d2d forgiveness + removed d2d termination** interaction with proximity curriculum starting at 0. Right now drones get zero d2d penalty AND zero proximity cost — there is literally no signal teaching separation, so "crashing into each other" may be a free action the policy exploits. Also nobody asked whether the 60-step cap is the env's `max_episode_length` or actual terminations — that distinction changes the entire diagnosis.

### Reviewer 4
**1. Strongest: B.** Only one with an executable Monday-order plan AND a concrete reward-math diagnosis: terminating=-25 over 60 steps = -0.42/step that the policy escapes by crashing faster. That's a falsifiable hypothesis, not vibes. C and E reach the same local-optimum conclusion but B gives you the knob to turn.

**2. Biggest blind spot: D.** Calls 60-step crashes "gold data." It's not — it's a degenerate distribution where the agent never reaches the regime LSTM+attention is supposed to model (opponent interaction near gates). Shipping league self-play over policies that die before passing one gate trains the encoder on noise. D also ignores that Geles24 explicitly bootstrapped from a working single-agent prior. "Release it" before gate_passed > 0.5/ep is premature optimization of the wrong axis.

**3. What ALL missed:** Nobody checked whether the -25 terminating penalty is being attributed to the right drone in the batched concatenated-state setup. With 4 drones sharing gradients and 10% d2d forgiveness, one drone's crash can poison the others' advantage estimates. Before tuning weights or arch, verify per-drone credit assignment in the PPO advantage computation — the reward scale debate is moot if the signal is being smeared across agents.

### Reviewer 5
**1. Strongest: B.** Only one with a concrete Monday plan + dimensional analysis. Catches the actual quantitative trap: `terminating=-25` over 60 steps = -0.42/step ambient cost, but `progress=+20` dense fires every step. Crashing fast is cheaper than flying badly. Also names the sparse-vs-dense scale mismatch (400 seen 1/50 eps vs 20 every step) with a fix. C and E reach similar conclusions; B operationalizes them with hours and knob values.

**2. Biggest blind spot: D.** Mistakes "stddev collapsing while drones die in 0.6s" for a platform-ready benchmark. Ignores that asymmetric AC + LSTM on a 4103-dim policy obs (4096 camera + 7 IMU, per the env cfg) with no working single-drone baseline = 2 weeks of plumbing on a broken substrate. "Ship the release" before the agent passes one gate is the textbook way to bake reward-hacking into a recurrent state you can't debug.

**3. All five missed:** Nobody questioned the **gate-pass detector itself**. `gate_passed ≈ 0.02/ep` could mean drones aren't passing OR `GateTargetingCommand`'s plane-crossing + bbox check is rejecting valid passes (false negatives = invisible reward). Also: with `env_spacing=0.0` all 4 drones share one track — collisions at spawn likely cause the 60-step deaths, independent of policy. Verify the sensor before tuning the reward.

---

## Council Verdict (Chairman)

### Where the Council Agrees
- **The 60-step episode length is the root signal, not architecture.** 4 of 5 advisors (Contrarian, First Principles, Outsider, Executor) name surviving 0.6s as the upstream problem. LSTM cannot help a policy that crashes before it can use memory.
- **Reward weights are misscaled for current episode length.** `gate_passed=+400` sparse and `progress=+20` dense, on 60-step episodes, means the dense term dominates ~12× per episode while sparse fires roughly 1 in 50 eps. Policy farms progress without orienting (lookat ≈ 0 confirms it).
- **Single-drone control experiment must come first.** All advisors except Expansionist explicitly say isolate the swarm variable by running single-drone NoCam-CTBR on the same reward stack. If single-drone also stalls at gate_passed ≈ 0, the bug is upstream of swarm.
- **Stddev collapse + positive reward is not convergence.** It's confident local-optimum lock-in — the policy commits to a bad strategy. Both Contrarian and Outsider call this out independently.

### Where the Council Clashes
- **Whether to ship LSTM + attention + league self-play now (Expansionist) or defer (everyone else).** Expansionist argues the swarm env is research-grade data, hardware advantage over Geles24 is real, and the LSTM unlock is strategic regardless of current gate-pass rate. Reviewers unanimously flag this as the council's biggest blind spot — Geles24 had a working single-agent flight prior; building recurrence over a non-functional substrate trains memory on noise.
- **Whether the fix is reward reshaping (Executor: crank gate_passed to +2000, decay progress near gate) or removing terminating penalty / lengthening episodes via curriculum (Contrarian).** Both target the same symptom; the difference is whether you trust the existing termination triggers.

### Blind Spots the Council Caught (emerged only in peer review)
1. **Reviewer 1:** The 60-step episode may not be a *learned* crash — it could be a spawn-time ContactSensor phantom. Recent commit history (`fix(asset): zero prop mass`, `bump contact force=80`) shows ongoing phantom-force fixes. If `illegal_contact` fires at step 0–5 from residual prop contacts, no reward tuning or architecture change fixes it.
2. **Reviewer 2:** Possible CTBR action / motor allocation / spawn-state bug. A 60-step death across all 4 drones with stddev collapsing is consistent with drones entering unrecoverable states at reset (prop spin-up transient, IMU warmup). Verify drone is physically flyable from t=0.
3. **Reviewer 3:** d2d termination is REMOVED and proximity curriculum starts at weight 0. Right now drones get **zero d2d signal at all**. "Crashing into each other" may be a free action. Also: is 60 = `max_episode_length` (env cap) or actual termination? The two are very different diagnoses.
4. **Reviewer 4:** In a batched concat-state PPO with 4 drones sharing gradients, one drone's -25 termination can poison the others' advantage estimates. Per-drone credit assignment is unverified.
5. **Reviewer 5:** `gate_passed = 0.02` could be a detector false-negative (`GateTargetingCommand`'s bbox check rejecting valid passes). With `env_spacing=0.0`, all 4 drones share one track at spawn — spawn collisions may explain the 60-step deaths independent of policy.

### The Recommendation
**Do NOT switch to LSTM PPO. Do NOT touch architecture.** Stay on MLP for the swarm. The architectural debate is premature optimization on a foundation the council does not yet know is sound.

The current symptom set (60-step episodes, gate_passed ≈ 0.02, lookat ≈ 0, stddev collapsing, total reward positive) is consistent with a **degenerate local optimum** caused by **reward miscalibration plus an unverified termination signal**. Five reviewers independently surfaced concrete suspect mechanisms (spawn phantom, CTBR mapping, env_spacing=0 spawn collisions, missing d2d signal, gate-pass detector false-negatives, batched advantage smearing). None of those are fixable by adding recurrence or attention.

The paper-style reward stack (proximity, ranking, L1 body rate, 10% non-terminal d2d) is structurally correct but operates on the wrong scale for current episode lengths. Fix the substrate first.

### The One Thing to Do First
**Run one diagnostic episode and instrument what's killing the drone at step 60.**

Concretely: `python3 scripts/rl/play.py --task Isaac-Drone-Racer-Swarm-NoCam-CTBR-Play-v0 --num_envs 1 --num_drones 4 --log 5` with a recent checkpoint, then print which `TerminationManager` term fires at the episode-ending step and at what step it fires. If termination fires at step 0–5, it's a spawn phantom (Reviewer 1's hypothesis) — fix contact thresholds and forget the reward debate. If it fires at step ~60 from `flyaway` or `gate_collision`, the policy is genuinely losing control — then tune rewards. If it never fires and 60 is `max_episode_length`, the diagnosis flips entirely (episodes are being truncated, not terminated).

Total time: ~30 minutes. Without this datum every downstream debate is uninstrumented.
