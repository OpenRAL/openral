# Which RoboCasa task drives the base while carrying?

**`DeliverStraw` does — 3.16–3.80 m of it, and it is already inside XR-1's
evaluated set.** This page records how that was measured, and why the obvious
way of answering the question gives the opposite, wrong answer.

## Why this page exists

[PR #143](https://github.com/OpenRAL/openral/pull/143) landed the Nav2 half of
[issue #108](https://github.com/OpenRAL/openral/issues/108): a dynamic footprint
published while an object is attached, and a lidar filter that removes the
payload's and the chassis's own returns. Both are load-bearing and **neither had
ever run against a base that translates mid-carry**, because no scene in this
tree did that — so a working footprint and a decorative one were
indistinguishable, and the deferred `CostCritic.consider_footprint` flip had
nothing to measure against.

`packages/openral_nav2_bringup/README.md` specified the scene that would close
#108 in four criteria and left one question open: is `loading_fridge` in XR-1
RoboCasa365's evaluated set? The answer turned out to matter less than the
question it was standing in for, which is simply **does any RoboCasa task make
the base drive while holding something.**

## Why this was measured and not read off the source

A first attempt answered "no, 0 of 50" by classifying task source with `ast`,
on the premise that RoboCasa's `ref=` keyword bounds a fixture pair to 0.10 m.
It was wrong, and it is written down here because the trap is in upstream's
docstring and will catch the next reader too. (It was caught before it shipped;
nothing in the tree ever carried the claim.)
`Kitchen.get_fixture`'s docstring does say that:

> `ref (str, Fixture, FixtureType)`: if specified, will search for fixture close
> to ref (**within 0.10m**)

**The docstring does not describe the code.** The body sorts candidates by
distance and keeps those within 0.10 m *of the nearest one*:

```python
dists = [OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures]
min_dist = np.min(dists)
close_fixtures = [f for (f, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10]
return self.rng.choice(close_fixtures)
```

That is a **tie-break among near-equidistant fixtures**, not a proximity bound.
A `ref=`'d fixture is the *nearest of its type*, and in a large kitchen the
nearest dining counter can be metres from the drawer the base starts at.

A second, independent error compounded it: linkage need not involve the pair
that matters. `DeliverStraw` registers `dining_counter` with `ref=self.stool` —
a *third* fixture — which says nothing about its distance from the drawer.

Both errors point the same way (they under-count), and both disappear once the
number is measured instead of inferred. The static classifier has been deleted
rather than patched; `tools/robocasa_carry_survey.py` builds the env.

## What is measured

For each task and seed: reset, then measure the straight-line distance from the
**base's start pose** to each object the task manipulates. A task requires a
mid-carry base translation when some object sits further than 1.0 m away — the
arm cannot bridge that, so the base must drive to reach it.

Distractors are excluded by name, and **both** RoboCasa spellings are needed —
`distr*` (117 uses) and `dstr*` (6). This is not cosmetic, and each spelling
caught a different false positive:

* `StoreLeftoversInBowl` has `distr1` / `distr2` on a far counter at ~5 m while
  every object it actually manipulates is within 0.94 m of the base.
* With only `distr` handled, `ArrangeBreadBasket` and `PanTransfer` reported
  `dstr_dining*` as their furthest object and read as cross-kitchen carries.

**Distance alone is a candidate filter, not a verdict.** It cannot tell a
carried object from a fixed one the arm reaches: `CloseFridge`'s only object is
`door_obj` at 0.84–1.06 m, which is a reach target attached to the fridge, not
a payload. So the tool reports *candidates*, and each one's `_check_success` has
to be read to confirm the object must be **moved** there. That reading is what
selected `DeliverStraw` below.

## The result

All 50 `target50` tasks, seeds 1–3, on `q-laptop` (RTX 5070 Laptop) through the
provisioned RoboCasa: **129 measurements over 43 tasks.** The remaining 7 —
`CloseBlenderLid`, `CloseToasterOvenDoor`, `NavigateKitchen`,
`OpenStandMixerHead`, `SlideDishwasherRack`, `TurnOnElectricKettle`,
`TurnOnSinkFaucet` — register no manipulable object at all, so there is nothing
to carry by construction. (`NavigateKitchen` is the interesting one: it *does*
drive the base across the kitchen, and refuses source/destination pairs closer
than 1.0 m, but it spawns no object — a live reset confirms `objects` and
`object_cfgs` are both empty. It navigates empty-handed.)

**11 tasks place an object beyond 1.0 m of the base start pose** (furthest
object per seed, in metres):

| task | set | seed 1 | seed 2 | seed 3 | furthest object |
| --- | --- | ---: | ---: | ---: | --- |
| `GatherTableware` | `composite_unseen` | 6.19 | **6.80** | 1.54 | `glass3` |
| `GarnishPancake` | `composite_unseen` | 3.37 | 3.76 | **4.01** | `pancake` |
| **`DeliverStraw`** | `composite_seen` | 3.16 | 3.61 | **3.80** | `glass_cup` |
| `GetToastedBread` | `composite_seen` | 2.96 | **3.48** | 2.17 | `plate` |
| `SetUpCuttingStation` | `composite_seen` | **3.33** | 1.45 | 0.60 | `receptacle` |
| `PackIdenticalLunches` | `composite_seen` | **2.95** | 2.06 | 2.73 | `tupperware0` |
| `PortionHotDogs` | `composite_unseen` | **2.43** | 0.91 | 1.37 | `plate1` |
| `SearingMeat` | `composite_seen` | 1.81 | 0.66 | **2.31** | `meat` |
| `MakeIceLemonade` | `composite_unseen` | 1.57 | 1.14 | **2.28** | `ice_bowl` |
| `RecycleBottlesByType` | `composite_unseen` | 1.03 | 1.01 | **1.08** | `bottle_glass2` |
| `CloseFridge` | `atomic_seen` | **1.06** | 0.84 | 0.92 | `door_obj` |

Every one of them registers its nearest object within 0.78 m — inside the
Panda's reach — so in each case there is something to pick up before the base
has to move.

Two things this table shows beyond the answer:

* **`CloseFridge` is the candidate filter's limit made visible.** Its only
  object is the fridge `door_obj`, which is attached to the fixture. Distance
  says 1.06 m; the task is a reach, not a carry. This is why each candidate's
  `_check_success` still has to be read.
* **Several tasks qualify at one seed and not another** — `SetUpCuttingStation`
  runs 3.33 / 1.45 / 0.60 m. Whether a run exercises the footprint at all is a
  property of the pinned seed, which is exactly what criterion 4 is for.

The two shortlisted after reading their success predicates are `DeliverStraw`
and `GetToastedBread`, both in `composite_seen` — i.e. inside **`target50`**,
the 50-task set XR-1's model card pins as its reference configuration (50
episodes per task, base seed 7, 1432/2500 = 57.28%). So the policy stays in
distribution; **no custom task is needed, and none was written.**

### `DeliverStraw`, the one that was pinned

```python
# robocasa/environments/kitchen/composite/serving_beverages/deliver_straw.py
self.drawer = self.register_fixture_ref("drawer", dict(id=FixtureType.TOP_DRAWER))
self.init_robot_base_ref = self.drawer
...
def _check_success(self):
    straw_in_glass_cup = OU.check_obj_in_receptacle(self, "straw", "glass_cup", th=0.5)
    gripper_far = OU.gripper_obj_far(self, obj_name="straw")
    return straw_in_glass_cup and gripper_far
```

The straw starts **in the drawer the base is parked at** (0.46–0.50 m, inside
the Panda's 0.855 m reach) and the glass cup sits on the dining counter beside a
stool, metres away. Success requires the straw *inside the cup*, so it must be
carried the whole distance. `scenes/deploy/robocasa_deliver_straw.yaml` pins
seed 3, the longest of the three seeds.

**It is not the longest carry in the set, and was not chosen for that.**
`GatherTableware` reaches 6.80 m and `GarnishPancake` 4.01 m. Both are
multi-object chains — `GatherTableware` succeeds on the mutual distances of
three glasses and a bowl, `GarnishPancake` on strawberry-on-pancake *and*
pancake-on-plate *and* plate-on-counter — so a failure there has several
explanations unrelated to the Nav2 footprint. `DeliverStraw` is the simplest
qualifying carry: one object, one destination, one `check_obj_in_receptacle`.
Fewest confounds between the carry and the verdict, because the verdict is what
#108 reads.

**Known precondition.** The drawer is closed at reset — measured,
`get_door_state` returns ~2.9e-08 — so the episode is open-drawer → grasp →
carry → place, and only the third phase exercises the footprint. A run that
never opens the drawer says nothing about Nav2 and must not be recorded as a
footprint failure. This is a property of the evaluated set rather than of this
choice: `GetToastedBread`, the runner-up, must start a toaster and wait for the
lever.

## Two things the same measurements settle about the other criteria

**Criterion 3 is unreachable for this robot.** The criterion asks for
"lidar-visible obstacles at the payload's height, so the scan filter's two
halves are both live". Measured, the carried object rides at **z ≈ 0.97–1.03 m**
(counter height), while `synthesize_laser_scan_2d` casts its fan at
`_LASER_DEFAULT_HEIGHT_M = 0.30`. **The payload never intersects the scan
plane**, so the payload half of `payload_scan_filter_node` cannot be exercised
by any counter-height carry — not by this scene and not by a different one. The
footprint publisher is unaffected: it ignores height by design ("Height is
ignored… the alternative is deciding on the robot's behalf which carried
geometry cannot hit anything"). What this scene exercises is the footprint and
the chassis self-filter, not the payload scan filter.

**Criterion 2 needs a polygon test, not a radius test.** Measuring the free
corridor at the scan plane (AABB occupancy at 2.5 cm, distance transform,
max-min bottleneck path) gives bottlenecks of **0.19–0.24 m** between the carry
endpoints — *below* the bare chassis's 0.444 m padded circumscribed radius, let
alone the 0.863 m carrying one. A circular test therefore says the robot does
not fit anywhere in a RoboCasa kitchen, including at its own start pose, which
is false: the chassis is a 0.70 × 0.50 m **rectangle** tucked against a counter,
and the counter is inside its circumscribed circle but outside the rectangle.
Criterion 2 as written ("an aperture the bare chassis clears but the
payload-grown polygon does not") can only be evaluated with an oriented-polygon
check — which is precisely the check `CostCritic.consider_footprint` turns on,
and precisely why the flag matters.

## Reproducing

```console
$ uv run python tools/robocasa_carry_survey.py --tasks DeliverStraw GetToastedBread --seeds 1 2 3
$ uv run python tools/robocasa_carry_survey.py --seeds 1 2 3          # all of target50
```

The pin itself is guarded by
`tests/sim/test_panda_mobile_hal_robocasa_carry.py`, which builds the scene
through `build_sim_env_from_yaml` and asserts both halves of criterion 1: the
straw inside the arm's reach, the cup beyond 1.0 m.

## Related

- `packages/openral_nav2_bringup/README.md` — the four criteria, the dynamic
  footprint, the scan self-filter, and the deferred
  `CostCritic.consider_footprint` flip this scene is the precondition for.
- [Sim environments](sim-environments.md) — how a `scenes/` entry reaches a
  RoboCasa env.
