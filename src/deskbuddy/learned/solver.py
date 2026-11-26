"""One-step proposal, geometric refinement, and explicit numerical fallback."""

import time

import numpy as np


def solve_goal(arm, start, target, config):
    tolerance, locked = config["ik_tolerance_m"], config.get("locked_joints", [])
    info = dict(requested=config.get("planner_mode", "numerical"), used="numerical")
    if info["requested"] == "flow_assisted":
        try:
            from .inference import get_proposal

            begun = time.perf_counter()
            method = config.get(
                "_proposal_method", config.get("proposal_method", "imf")
            )
            info["proposal_method"] = method
            model = get_proposal(
                config["flow_checkpoint"],
                config,
                method=method,
            )
            info["training_config_changed"] = not getattr(
                model, "configuration_matches", True
            )
            strategy = config.get("flow_seed_strategy", "direct")
            inputs = [("direct", target)] if strategy != "workspace_affine" else []
            if strategy in ("workspace_affine", "best"):
                from .coordinates import map_workspace_target

                training = getattr(model, "training_config", None)
                if training is not None:
                    inputs.append(
                        (
                            "workspace_affine",
                            map_workspace_target(target, config, training),
                        )
                    )
                elif strategy == "workspace_affine":
                    raise ValueError(
                        "Checkpoint has no workspace metadata for affine seeding"
                    )
            candidates = []
            for name, input_target in inputs:
                raw = model.predict(start, input_target)
                arm.frames(raw)  # Finite, well-formed guesses only.
                bounded = np.clip(raw, arm.lower, arm.upper)
                gap = float(np.linalg.norm(arm.fk(bounded) - target))
                candidates.append((gap, name, input_target, raw, bounded))
            info["proposal_ms"] = (time.perf_counter() - begun) * 1000
            # An initial guess is never a motor goal. Even a poor finite guess
            # can be projected and repaired by bounded IK; quality gates belong
            # on the refined solution and complete path, not on the raw seed.
            info["candidate_seed_gaps_mm"] = {c[1]: c[0] * 1000 for c in candidates}
            begun = time.perf_counter()
            failures = []
            for gap, name, input_target, raw, bounded in sorted(
                candidates, key=lambda c: c[0]
            ):
                try:
                    goal = arm.ik(
                        target, start, tolerance, locked, initial_guess=bounded
                    )
                except ValueError as exc:
                    failures.append(f"{name}: {exc}")
                    continue
                info.update(
                    used="flow_assisted",
                    seed_strategy=name,
                    refinement_ms=(time.perf_counter() - begun) * 1000,
                    raw_model_gap_mm=float(np.linalg.norm(arm.fk(raw) - target)) * 1000,
                    refinement_seed_gap_mm=gap * 1000,
                )
                adjustment = np.rad2deg(bounded - raw)
                if np.any(adjustment):
                    info["seed_projection_deg"] = adjustment.tolist()
                if hasattr(model, "in_training_domain"):
                    info["outside_training_coverage"] = not model.in_training_domain(
                        start, input_target
                    )
                if name == "workspace_affine":
                    info["mapped_proposal_target_m"] = input_target.tolist()
                return goal, info
            raise ValueError("; ".join(failures))
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            info["fallback_reason"] = str(exc)
    return arm.ik(target, start, tolerance, locked), info


def planner_status(diagnostics, requested="numerical"):
    """Only report actual numerical fallback; normal planner use stays quiet."""
    messages = []
    for index, info in enumerate(diagnostics or []):
        if info.get("used") == "numerical" and (
            info.get("requested", requested) == "flow_assisted"
        ):
            prefix = f"Segment {index + 1}: " if len(diagnostics) > 1 else ""
            messages.append(
                prefix
                + "Numerical fallback: "
                + info.get("fallback_reason", "learned solver unavailable")
            )
    return " | ".join(messages)
