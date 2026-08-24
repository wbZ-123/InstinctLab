# No-Fly Penalty Design

## Goal

Discourage hopping by penalizing a control step only when neither foot has
ground contact. Normal single-support walking and swing-foot lift-off must not
be penalized.

## Reward Semantics

The reward function reads the existing contact-force sensor for the two ankle
roll links and applies the same 1 N contact threshold already used by the
parkour foot-contact rewards.

For each environment and control step:

```text
left foot contact OR right foot contact  -> 0
left foot airborne AND right foot airborne -> 1
```

The parkour reward configuration assigns this indicator a weight of `-1.0`,
so every double-flight control step contributes `-1.0` before the environment
time-step scaling performed by the reward manager. Longer double-flight
intervals therefore accumulate a larger penalty.

The term remains active during Recovery. It does not replace `feet_air_time`:
that reward still encourages useful single-support steps, while `no_fly` only
discourages simultaneous loss of both contacts.

## Scope

- Add one small reward function to the existing parkour reward module.
- Add one `RewTerm` named `no_fly` to `RewardsCfg` with weight `-1.0`.
- Reuse the existing contact sensor and ankle body selection.
- Do not add a grace time, command gate, terrain-specific condition, or new
  sensor.
- Do not change Recovery transitions or the existing `feet_air_time` reward.

## Verification

Unit tests must demonstrate:

1. Two airborne feet return `1` before weighting.
2. Left-only, right-only, and double contact return `0`.
3. The parkour configuration registers `no_fly` with weight `-1.0` and the
   two ankle links.
