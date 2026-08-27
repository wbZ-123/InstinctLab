# Planner reward expression backup (before margin/step update)

Captured from the working tree before the 2026-08-26 planner reward update.

Working-tree commit at capture: `e5c0836` (`tajdoi`). The source reward file was
already modified relative to that commit; this backup records the exact current
implementation used for comparison.

## Mounted reward term

`parkour_env_cfg.py` mounted one planner term with weight `1.0`:

```python
learned_foothold_planning = RewTerm(
    func=mdp.learned_foothold_planning_event_reward,
    weight=1.0,
    params={
        "sensor_name": "foothold_planner",
        "reachability_radius_x": _DEFAULT_FLAT_PROVIDER_CFG.outer_radius_x,
        "reachability_radius_y": _DEFAULT_FLAT_PROVIDER_CFG.outer_radius_y,
        "velocity_lookahead_s": _DEFAULT_FLAT_PROVIDER_CFG.velocity_lookahead_s,
        "nominal_step_width_m": _DEFAULT_FLAT_PROVIDER_CFG.nominal_step_width,
        "velocity_std": 0.5,
    },
)
```

## Previous helper and event reward

The previous reward used `_nominal_deviation_reward` with a 2 cm tolerance,
`_signed_command_progress_score`, the existing sole-penetration safety score,
and this branch:

```text
execution/preflight invalid -> -1
geometry invalid -> -1
learned point unsafe -> learned sole-penetration score
learned point safe and nominal safe -> nominal deviation reward
learned point safe and nominal unsafe -> signed command progress - deviation cost
non-event -> 0
```

The previous implementation did not contain a signed clearance margin outside
the cylinders, did not penalize oversize forward steps separately, and did not
normalize the final sum as a four-component reward. It also had no terrain
tread component; this update intentionally continues to omit that component.

See the pre-update source file in the same worktree for the exact complete
function bodies.
