# Object-centric geometry and pose: integration survey

Research date: **2026-10-19**. Inspected checkout: `feat/418-object-tracking`,
`85a8c8128`. This is a design recommendation, not an implemented integration or a
performance result. No dependencies were installed and no simulations or tests were run.

## Recommendation

**Keep OpenRAL's world occupancy and safety interfaces. Add persistent geometry in
each object's own frame, with a separately observed pose.** Start with bounded,
segmented point clouds and PCL registration; evaluate Open3D TSDF fusion only if
the point-cloud model lacks useful surface coverage. PCL registration here is a
proposed addition, not a capability attributed to the current tracker.

For a known mesh, **Isaac ROS FoundationPose is the strongest packaged ROS 2 pose
candidate** in this shortlist. NVIDIA's newer Apache inference library is also
worth a pilot, especially for pose from reference views without CAD, but needs a
custom ROS integration. Pin the runtime and model artifact separately. These
supply pose estimates; OpenRAL must still handle identity, observation age,
model ownership, uncertainty and safety geometry. For unknown objects, none of the
reviewed packages supplies a complete, production-ready Jazzy replacement for all
those responsibilities. Details and source evidence follow.

**Do not choose nvblox merely because it supports dynamic scenes.** Its documented
dynamic mode maintains static geometry and decaying dynamic occupancy. Persistent
shape reconstruction for a particular moving object is an additional system.

### Four outputs that must remain distinguishable

| Output | Meaning | What it cannot establish alone |
| --- | --- | --- |
| Object identity | Which instance an observation belongs to | Shape, correct pose or permission to touch it |
| Observed pose | Object-to-world transform at a sensor timestamp | Hidden geometry, unchanged grasp or containment |
| Reconstruction | Accumulated visible surfaces in object coordinates | That unseen space contains no part of the object |
| Safety envelope | Conservative geometry accepted under explicit rules | Contact permission or force protection |

Declared dimensions or a CAD mesh can supply a **prior on complete shape**. They
still need correct instance association, units, frame, scale, revision and
tolerances. An ordinary SKU nominal dimension is not automatically a validated
maximum dimension. A search box constrains where to look. A workspace declaration
scopes manipulation and any separately approved contact rules. Neither measures
the object's size.

## What is already in OpenRAL

The current tracker combines recent clear camera fits in a short window and uses
heuristics for hidden extent; it is not a persistent object-local surface mapper.
Binding constructs carried geometry, and subsequent protrusion observations can
grow it. See [tracker.py](../../python/world_state/src/openral_world_state/object_tracker/tracker.py)
(`VIEW_WINDOW_S`, multi-view fusion, measured hidden size) and
[binding.py](../../python/world_state/src/openral_world_state/object_tracker/binding.py)
(binding construction and `grow_by_protrusion`).

The world-map bridge already handles tracked-object current, swept, vacated and
released cells, with identity checks and support/static-geometry exclusions.
Its documentation records residual hazards, including observations changing
between the bridge snapshot and an octree clear. This is valuable existing work;
it is not yet an object-owned voxel contribution ledger. See
[identity clearing and OT-8](../../packages/openral_octomap_bridge/README.md#and-a-tracked-object-is-cleared-by-identity-issue-418).

The safety kernel's **K4 union rule prevents shrinkage within a binding epoch**.
Refining the perception model does not authorize reducing the held collision
envelope. The agreed direction is to tighten before binding; shrinking during a
carry requires a later, explicitly reviewed revalidation rule. Changing the epoch
to evade the union is not that rule. See
[K4 contract](../../cpp/openral_safety_kernel/README.md) and
[engineering safety requirements](../../CLAUDE.md#3-architecture-discipline).

The repository's ROS CI uses Jazzy. `sim run` steps a policy and simulator directly;
`deploy sim` composes the ROS perception and safety path. Keep `sim run` as the
policy benchmark and use `deploy sim` to evaluate this production-facing design.
Equal task success does not establish equal safety behavior. Sources:
[ROS CI](../../.github/workflows/test-ros2.yml),
[sim runner](../../python/sim/src/openral_sim/sim_runner.py),
[deploy composition](../../packages/openral_rskill_ros/launch/deploy_e2e.launch.py).

## Practical shortlist

### 1. Existing OctoMap plus proposed PCL object registration

**Best first integration: CPU, limited new machinery.** OctoMap supplies probabilistic
occupied/free/unknown mapping and ray insertion. PCL supplies point-cloud filtering
and registration, including ICP/GICP and initial alignment methods. These are
building blocks: instance association, persistent object-local models and moving
object ownership still belong in OpenRAL. Local registration needs a suitable
initial alignment and observable geometry; convergence is not an identity or
containment proof. Sources: [OctoMap overview](https://octomap.github.io/octomap/doc/index.html),
[ray insertion API](https://octomap.github.io/octomap/doc/classoctomap_1_1OccupancyOcTreeBase.html),
[PCL registration](https://pointclouds.org/documentation/group__registration.html).

Jazzy's official distribution contains `octomap`, `octomap_server`, `pcl_ros` and
`pcl_conversions`. Those ROS packages do not constitute a moving-object mapper.
The core OctoMap and PCL libraries use BSD licenses. CPU deployment avoids a CUDA
requirement; ARM64 target build and timing still need validation, not an inference
from desktop results. Sources: [Jazzy distribution](https://github.com/ros/rosdistro/blob/master/jazzy/distribution.yaml),
[OctoMap core license](https://github.com/OctoMap/octomap/blob/devel/octomap/LICENSE.txt),
[PCL license](https://github.com/PointCloudLibrary/pcl/blob/master/LICENSE.txt).

Suggested initial model: a bounded object-frame point set with observation
provenance and coverage. A small object-local OctoMap is an alternative when ray
evidence matters. Merely storing XYZ points or surfels does not provide an
explicit free/unknown volume; keep that distinction in the output contract.

### 2. Open3D for an optional object-local TSDF

Open3D provides CPU/CUDA TSDF integration from calibrated depth and known camera
poses, plus surface point and mesh extraction. It is a suitable library for
custom per-object fusion once masks and object poses are available. Its scene
reconstruction examples do not supply OpenRAL's identity lifecycle or safe moving
object clearing. Sources: [TSDF integration](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html),
[MIT license](https://github.com/isl-org/Open3D/blob/main/LICENSE).

Open3D 0.19 documents **experimental ARM64 support**. Its ARM64 Python wheels lack
CUDA; Jetson CUDA requires a source build. A ROS `open3d_conversions` entry exists
in Jazzy's distribution, but conversion support is not an object-mapping node.
Treat the lifecycle node and mapping policy as custom integration. Sources:
[ARM support](https://www.open3d.org/docs/release/arm.html),
[Jazzy distribution](https://github.com/ros/rosdistro/blob/master/jazzy/distribution.yaml).

For the proposed design, give each object its own TSDF, store valid-observation
weights, and retain a bounded keyframe history. Pose corrections require rebuilding
affected contributions or a deliberately implemented reversible integrator. Do
not assume a weighted TSDF can undo arbitrary past observations. Unobserved voxels
remain unknown; a smooth extracted surface or positive TSDF sample is not a
certificate that a whole collision volume is free.

### 3. Isaac ROS FoundationPose for known-mesh pose

The released ROS node takes a mesh, aligned RGB/depth, camera calibration and a
segmentation mask for pose estimation. The tracking node additionally consumes
the prior pose. Its API exposes symmetry handling and tracking reset controls.
These are useful operational facilities, not calibrated safety guarantees.
The inspected ROS interface is **mesh-based**; the research paper's model-free
reference-view mode must not be attributed to that interface. Sources:
[FoundationPose ROS API, release 4.6](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html),
[FoundationPose paper](https://arxiv.org/abs/2312.08344).

Version selection matters: the current latest Isaac ROS documentation names
**Lyrical**; the release-4.6 archive explicitly names **Jazzy** and NVIDIA GPU
platforms, including Jetson and DGX Spark. Use a pinned compatible image/package
set, rather than following `latest`. This is an NVIDIA GPU route, not the CPU
baseline, and generic ARM64 compatibility does not establish compatibility with
every ARM64 computer. Sources:
[current pose-estimation platform requirements](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/index.html),
[release-4.6 Jazzy/platform requirements](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_nvblox/index.html).

The reviewed FoundationPose artifacts have **different license surfaces**:

| Artifact | Verified terms | Integration consequence |
| --- | --- | --- |
| Isaac ROS pose-estimation wrapper | Apache-2.0 | Does not determine model or SDK terms |
| NVlabs FoundationPose research repository | NVIDIA license §3.3 limits use to noncommercial research/evaluation | Do not copy this implementation into the production Apache codebase |
| NGC FoundationPose model | Separate Deep Learning Models EULA | GPU restriction and distribution conditions remain; pin and record the selected model's terms |
| New NVIDIA FoundationPose Inference Library | Apache-2.0 main library | Custom C ABI/Python integration; optional renderer and SDK dependencies have their own terms |
| Current Hugging Face `nvidia/foundationpose` weights | NVIDIA Open Model License | Commercial use permitted under that license; do not apply the older NGC card's terms to this artifact |

Sources: [ROS wrapper license](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/main/LICENSE),
[research-code license](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE),
[NGC model card](https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/models/foundationpose/-),
[model EULA §§1–2](https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf),
[new inference-library license](https://github.com/nvidia-isaac/foundation-pose-inference-library/blob/main/LICENSE),
[Hugging Face model card](https://huggingface.co/nvidia/foundationpose),
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
The model EULA grants use and conditional application distribution, restricts
applications to NVIDIA GPU systems, and disallows standalone model distribution.
Do not collapse these terms into either “everything is Apache” or “all FoundationPose
implementations are noncommercial.”

**New inference-library option.** The official
[NVIDIA FoundationPose Inference Library](https://github.com/nvidia-isaac/foundation-pose-inference-library)
supports CAD meshes or RGBD-mask reference views, with registration and tracking
through C ABI/Python. Its published platform table lists x86 NVIDIA GPUs, validated
on RTX PRO 6000 Blackwell, and Jetson AGX Thor; it does not explicitly establish DGX
Spark support. Its documented container uses CUDA 13.2/TensorRT 10.16. CPU-memory
inputs are supported, but computation remains GPU-based. No packaged Jazzy node is
established by these interfaces; “ROS integrable” means wrapper work here.

This is a more promising production-code starting point than copying the NVlabs
research implementation. Model-free reference views provide an object-specific
seed; they do not amount to verified continuous shape reconstruction of a wholly
unseen object. The API returns pose, not a conservative geometry bound. Its header
explicitly says the score comes from ScoreNet during registration, while tracking
sets it to `1`; that value cannot establish tracking confidence. Independent
residual, visibility, observation-age and slip checks remain necessary. Sources:
[runtime architecture](https://github.com/nvidia-isaac/foundation-pose-inference-library/blob/main/doc/ARCHITECTURE.md),
[C API output contract](https://github.com/nvidia-isaac/foundation-pose-inference-library/blob/main/include/foundation_pose_nvidia/c_api.h).
The default renderer avoids nvdiffrast; that optional plugin, CUDA/TensorRT,
container and downloaded weights retain their separate terms. Source:
[third-party notices](https://github.com/nvidia-isaac/foundation-pose-inference-library/blob/main/THIRD_PARTY_NOTICES.md).

### nvblox: useful world-map candidate, lower priority for this problem

nvblox is GPU-accelerated depth/pose-based mapping with TSDF, occupancy and ESDF
outputs. Isaac ROS supplies an actual ROS integration. The core and ROS wrapper
are Apache-2.0, with retained BSD notices for relevant core source. Sources:
[nvblox ROS overview](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_nvblox/index.html),
[core license](https://github.com/nvidia-isaac/nvblox/blob/public/LICENSE.md),
[ROS license](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/blob/main/LICENSE).

Its documented dynamic mode splits observations into a static TSDF/freespace
mapper and dynamic occupancy mapper. Decay represents loss of knowledge of
unobserved dynamic regions; it does not identify a banana, preserve its shape,
or relocate its local reconstruction. Unknown-distance outputs and decay options
also require explicit mapping into OpenRAL's semantics. Source:
[release-4.6 mapping modes and decay parameters](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/api/parameters.html).

Consider it if profiling later establishes world-map throughput as the bottleneck,
or if an ESDF consumer is needed. Replacing OctoMap now would add GPU/platform and
map-semantics work without supplying object identity, object-local reconstruction,
or a conservative held envelope.

### Research references and alternatives outside the first pilot

| Candidate | Relevant capability | Why it is not the first deployment choice |
| --- | --- | --- |
| BundleSDF | Joint RGBD tracking and neural reconstruction of an unknown rigid object; concurrent pose/shape optimization | Research implementation, custom integration; noncommercial research/evaluation license. Online mask tracking is an additional dependency. No verified shipped Jazzy/ARM64 component in the reviewed release. |
| Fusion++ | Separate per-object TSDFs and object pose graph | Useful representation reference, but the paper explicitly assumes a static environment and does not track individually moving objects. No verified production Jazzy package/license route established here. |
| Voxblox | CPU TSDF/ESDF world reconstruction | BSD-3-Clause; reviewed ROS wrapper uses ROS 1 catkin. Porting it still leaves moving-object identity and geometry ownership to implement. |
| Kimera / Kimera-Semantics | Metric-semantic scene reconstruction built on Voxblox | Reviewed Semantics integration uses ROS 1 catkin/Melodic and BSD terms. A semantic scene mesh is not by itself persistent reconstruction of a manipulated object. |

Sources: [BundleSDF paper](https://arxiv.org/abs/2303.14158),
[BundleSDF implementation](https://github.com/NVlabs/BundleSDF),
[BundleSDF license §3.3](https://github.com/NVlabs/BundleSDF/blob/master/LICENSE.txt),
[Fusion++ §1–3](https://arxiv.org/html/1808.08378),
[Voxblox](https://github.com/ethz-asl/voxblox),
[Voxblox ROS package](https://github.com/ethz-asl/voxblox/blob/master/voxblox_ros/package.xml),
[Kimera-Semantics installation](https://github.com/MIT-SPARK/Kimera-Semantics#1-installation),
[Kimera-Semantics license](https://github.com/MIT-SPARK/Kimera-Semantics/blob/master/LICENSE.BSD).

## Proposed integration contract

The following is a design proposal, not a claim about existing APIs.

1. **Input:** timestamped RGBD/mask, calibration and TF at acquisition time;
   segmentation identity; optional externally supplied mesh or declared dimensions.
   Store the declaration's source, hash/revision, units, object frame and tolerance.
   Load external USD geometry/annotations at runtime when available. Keep those
   assets external; do not create copies of private scene assets in this repository.
2. **Object state:** stable instance ID plus lifecycle generation, local model,
   observed world pose, pose quality/ambiguity, geometry revision, coverage and last
   contributing timestamps. A known mesh and measured surface evidence remain
   distinguishable. CAD supplies shape initialization. An explicitly declared fixed
   location can supply a pose prior; continued tracking of movable objects comes
   from sensors, never privileged live simulator object poses.
3. **Fusion:** transform each accepted camera point into the object's frame using
   the observed object pose and timestamped camera TF. Moving a rigid object then
   changes its world transform without dragging an old world-space surface trail.
   Fuse only observations consistent with identity, depth, pose and visibility;
   stop fusion on disagreement to avoid permanently corrupting the model.
4. **World occupancy:** keep object contributions separate from static/background
   evidence. Exclude confidently assigned object returns from static fusion only
   when an equally conservative object layer covers them; ambiguous returns remain
   obstacles. If evidence was already fused, retain ownership/history sufficient
   to remove only that object's contribution. A new object pose must never erase a
   support or neighboring object that shared a voxel.
5. **Vacated space:** retiring an object's old contribution does not itself observe
   free space. Preserve other evidence; otherwise represent the location as unknown
   until valid rays establish free space. Unknown must remain conservative at the
   safety boundary. This is a proposed stronger contract than the present clearing
   heuristic and requires review, not a reinterpretation of its existing behavior.
6. **Safety export:** publish a bounded conservative primitive/hull representation
   separately from the detailed model, with identity, timestamp, frame, revision,
   provenance and validity. A convex hull encloses measured points, not necessarily
   the whole object. Hidden extent needs an accepted prior or a conservative unknown
   bound. Inflate for pose, depth/calibration, discretization and observation age
   under an explicit error model; an arbitrary confidence score is insufficient.

### Rigidity, symmetry and failure handling

These requirements follow from the proposed representation:

- **Rigid motion is not deformation.** A rigid banana turning or slipping changes
  pose. A bending object invalidates the rigid model assumption. Detect persistent
  inconsistency, preserve a conservative envelope and stop using the model for
  tightening; nonrigid reconstruction is a separate scope.
- **Symmetry limits observability.** A nearly spherical orange may give a good
  center/radius with poorly determined rotation. Keep equivalent orientations or
  a symmetry-invariant collision envelope. Smooth-looking orientation is not
  evidence that rotation is known.
- **Occlusion is not disappearance.** Stop geometry fusion when identity/pose is
  ambiguous. Retain an uncertainty region and mark it stale according to a defined
  deadline. Reappearance needs association checks; do not silently relabel a nearby
  same-class object as the previous instance.
- **Held prediction is not observed pose.** Hand kinematics can predict motion,
  but visible evidence must check slip. A discrepancy must not be hidden by forcing
  the reconstruction back onto the predicted attachment transform.
- **Bound work and age.** Cap points/voxels, keyframes, active objects, registration
  iterations and queued frames. Prefer the newest complete synchronized input over
  processing a stale backlog. Log drops, queue delay, processing time, output age,
  pose resets and fusion rejection. Never restamp old geometry as newly observed.
- **Separate fast pose from slower shape.** Publish fresh pose even when mesh
  extraction or reconstruction is delayed, retaining the last valid shape and its
  actual age. Deadline failure follows the existing stale-data safety policy.

The selected algorithms have no established end-to-end latency guarantee on
OpenRAL's deployment hardware. NVIDIA's model card profiles TensorRT inference,
not the whole ROS graph; nvblox's published reconstruction timings use specified
datasets and 5 cm voxels. Neither establishes millimeter object reconstruction
latency with segmentation, TF, transfers and safety export included. Sources:
[FoundationPose performance scope](https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/models/foundationpose/-),
[nvblox benchmark scope](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_nvblox/index.html).

## Phased evaluation using public/example assets

**Phase A — instrument the current path.** Preserve current binding/safety behavior.
Record observation age, model containment and map trails for one sphere/orange,
one curved banana and one generic box. Use public/example geometry with recorded
asset licenses and versions, or simple generated geometry; use no private assets.

**Phase B — shadow object-local models.** Compare PCL registration plus bounded
points against an Open3D TSDF only where additional surface fusion is useful.
Evaluate the packaged ROS FoundationPose path for known meshes and the new
inference library for known meshes/reference views on compatible GPUs. Shadow
outputs must not quietly replace safety geometry.

**Phase C — reviewed pre-bind geometry and occupancy ownership.** Introduce typed
contracts and recorded Safety-WG rules for declared priors, conservative export,
staleness, ambiguity and clearing. Require evidence of containment before adopting
tighter bounds. Keep K4 during carry. A held-shrink revalidation mechanism is a
later change with its own safety case.

### Paired scenario matrix

Run the following as proposed evaluation cases, using the same assets, task intent,
initial conditions, policy revision and perturbation seed where both runners support
them. Record unsupported combinations rather than treating them as passing parity.
These cases have not been run or added as tests.

| Case family | Declaration pair | Perturbation and expected evidence |
| --- | --- | --- |
| Sphere/orange | Known dimensions/mesh vs no shape declaration | Translate, rotate, occlude and reappear. Center/radius containment matters; symmetric rotation must not create false precision. |
| Curved banana | Correct full mesh vs dimensions only vs unknown shape | Rotate to expose unseen curvature, partially hide the ends, move while free, grasp and slip. A small visible fit must not silently truncate hidden geometry. |
| Generic box | Correct dimension intervals vs omitted dimensions | View one face then several faces; move near a support and neighbor. Measure pre-bind tightening and retained unknown extent. |
| Wrong prior | Correct catalog record vs wrong size/unit/axis/revision or swapped SKU | Reject or quarantine the prior on conflicting evidence. Include an undersized prior whose missing part is initially occluded; visible agreement alone cannot validate it. |
| Search/workspace metadata | Correct search box/workspace vs omitted; separately wrong/too small | Place the object outside the search prior and near a workspace boundary. A failed search must remain explicit; declarations must not delete obstacles. |
| Identity ambiguity | One object vs two similar objects | Swap/move one during occlusion, merge/split masks and reappear. Measure ID switches, false re-association and collateral occupancy removal. |
| Contacts and map ownership | Allowed support contact vs forbidden neighbor/fixture/body contact | Move/rotate the object beside thin supports and another object; cover then uncover old occupancy. No contact permission may follow merely from shared voxels or matching geometry. |
| Sensor/compute failures | Nominal stream vs delayed/dropped/skewed frames and overloaded fusion | Measure conservative stale behavior, pose age, queue bounds, recovery and persistent memory growth. |

Use `sim run` for task/policy performance. Use `deploy sim` for perception,
occupancy, binding and kernel outcomes. Simulator truth may score pose, shape
containment, missed collision and false clearance in the evaluator. Privileged
live object poses, segmentation IDs and collision truth must not feed the runtime
perception path. Explicitly declared asset geometry and fixed-location priors are
allowed inputs under the same validation contract as a real deployment catalog;
they must be distinguished from evaluator-only ground truth.

### Concrete cases across the three simulations

Each row below is a proposed family of paired runs, not an executed result. Keep
private restock stages, assets and scenario definitions in their external project;
the public repository needs only generic interfaces and public tutorial fixtures.

| Simulation | Declaration and observation cases | Motion, contact and failure cases |
| --- | --- | --- |
| SO-101 PickOrange | Declared orange diameter interval versus unknown size; correct versus wrong diameter; known shape with unknown location; declared search box with orange inside/outside; three similar oranges with a specific-instance versus any-orange task | Roll after first sight; partial occlusion and reappearance; merged masks; gripper contact with intended orange versus another object; table contact remains protected; stale depth while the orange moves |
| DROID banana to bowl | Full banana mesh versus dimensions only versus unknown shape; hidden tip revealed from a new view; wrong mesh revision; bowl declared as a workspace versus no workspace; declared bowl position versus observed localization | Banana slips during carry; bowl is pushed after localization; banana tip contacts rim; gripper versus table contact; bowl hollow interior must not be filled by a bounding-box assumption; reflective or missing bowl depth is unknown |
| OpenArm restock | SKU dimensions with free localization versus exact declared initial slots; any-carton versus named-SKU identity; dimensions absent/wrong/swapped; tote workspace with/without an approved contact request; scene units/scale mismatch | Push carton and update pose; two cartons touch and masks merge; tote moves after confirmation; gripper-to-carton, gripper-to-tote and carried-carton-to-tote pairs; wrist/forearm negative probes unless explicitly approved; effort absent/stale/saturated; thin shelf support shares cells with a moved carton |

Cross the meaningful cases with valid, delayed and missing sensor data, and with
declarations present and absent. Do not use a full Cartesian product blindly:
pair baseline cases first, then combine the failure modes most likely to interact.

## External declarations and contradiction handling

This section proposes a contract; the generic USD annotation importer and these
mismatch checks are not implemented by this report.

**Ownership.** Keep restock meshes, SKU records, workspace/search declarations and
task-specific contact requests in the external USD stage or its referenced USD
layers. Simulation YAML references that stage. A possible documented namespace is
`/OpenRAL/Catalog` for shared records, with relationships from object and workspace
prims to their records; the exact paths are a schema decision, not an existing
OpenRAL convention. Import into typed runtime records without committing copies
of private assets or records. USD metadata supports application data, but its
presence alone does not validate OpenRAL semantics. Sources:
[USD application metadata](https://openusd.org/release/api/class_usd_object.html),
[USD layers](https://openusd.org/release/glossary.html#layer).

Record whether each declaration is a nominal value, interval, maximum extent,
solid shape, search region or fixed/movable pose prior. Include source revision,
units, frame and tolerances. Prefer that declared information over free-form
guesses, while preserving sensor evidence that disputes it. A search region is
not a claim that everything outside it is empty. A declared workspace alone is
not confirmed contact geometry or force permission.

**Contradiction rule.** A credible, correctly associated observation conflicts
when no pose and shape allowed by the declaration and calibrated uncertainty can
explain it. Compare in the object's frame, accounting for timestamped TF, depth,
mask boundaries, registration uncertainty and discretization. Require coherent
support appropriate to the sensor; do not invent a universal millimeter cutoff
or count repeated correlated frames as independent proof. These error budgets
must be established before using a prior to tighten safety geometry.

| Observation | Interpretation and response |
| --- | --- |
| Coherent orange surface span exceeds a declared maximum diameter plus the validated error budget | Geometric conflict: suspend reliance on that maximum and reobserve identity/size/calibration |
| Carton fits its declared dimensions after a permitted rotation/translation | Pose update, not a size contradiction |
| Visible carton occupies less than a declared maximum | Compatible; a maximum is not an exact size, and occlusion may hide the remainder |
| Two cartons merge into one mask | Association ambiguity; do not enlarge one SKU model to absorb both |
| Banana tip is occluded or depth is invalid | Unknown geometry; absence of returns does not justify shrinkage |
| Valid rays traverse material a registered solid model requires to exist | Possible negative contradiction, only with a suitable material/sensor model and explicit free-space evidence |
| Fixed tote moves after confirmation | Workspace/pose mismatch: re-localize and revalidate the contact region |
| All objects exhibit the same apparent displacement | Suspect camera calibration, TF or synchronization before classifying every catalog entry as wrong |

The present binary safety-grid representation does not distinguish all free and
unobserved cells, so a zero voxel is insufficient for the negative-evidence test.
A conservative fitted box larger than a catalog box is also insufficient: use
actual surface evidence, not the tracker's own hidden-extent heuristic.

On a credible conflict, immediately suspend the affected prior-based tightening
and contact exemption. Retain raw obstacle evidence and a valid broader envelope;
hold and reobserve if none is available. Persistence helps diagnose outliers,
wrong identity or calibration, but must not leave an apparently undersized safety
envelope active while waiting for more frames. Resolving a conflict can recover a
prior only through an explicit revalidation step with recorded evidence.

Report at least: identity errors; pose error modulo valid symmetries; maximum
geometry protrusion beyond the exported envelope; useful tightening before bind;
unknown-space coverage; incorrect clearing of other objects/supports; stale world
trails; task success and refusals by reason; end-to-end observation age median/p95/
worst; deadline misses; CPU/GPU use; peak and steady-state memory. Set acceptance
limits from the robot's motion/error budget before running, rather than inventing
a universal frame-rate target.

## Safety limits that mapping cannot resolve

A tighter geometry model does not authorize contact, establish a safe contact
force, or substitute for an effort monitor. At the inspected checkout, commanded
`JOINT_TORQUE` validation is separate from measured-effort enforcement; the generic
force fields are documented as unenforced, and placement contact-force handling
does not establish a general measured joint-effort cutoff. Audit that path as a
separate safety capability before relying on it. Sources:
[command validator](../../cpp/openral_safety_kernel/src/validator.cpp),
[lifecycle kernel](../../cpp/openral_safety_kernel/src/lifecycle_kernel.cpp),
[force schema contract](../../python/core/src/openral_core/schemas.py),
[collision checks](../../cpp/openral_safety_kernel/src/collision.cpp).

**Contact proposal.** Keep calibrated robot/unit hard ceilings in robot safety
configuration. An external USD task may request a lower, pair-specific operating
budget; it cannot raise the robot ceiling. USD drive `maxForce` limits actuator
drive force/torque and does not define a safe incidental-contact threshold. Raw
joint effort includes effects other than external contact; a calibrated external
torque estimate or force sensor requires its own validity and freshness checks.
Sources: [USD drive force limit](https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html),
[Franka contact/collision thresholds](https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html).

Define light contact by approved body pairs, a calibrated low force/torque budget,
fresh valid feedback, approach-speed limits and response margin below the hard
cutoff. Absence of feedback is unknown, not zero force. Low speed or small simulated
penetration alone cannot establish light contact. A simulator may expose a clearly
identified contact-force channel for evaluation; this is not evidence that the
real robot has an equivalent sensor. The user requested gripper-to-tote,
gripper-to-carton and carried-carton-to-tote pairs. Wrist contact remains undecided;
other robot parts remain protected. Pair permissions do not confer SKU identity or
authorize a wrong-object grasp. The same raw joint-torque number does not imply the
same contact force at different lever arms or with different carried loads.

The remaining decisions are the accepted prior-validation rule, unknown-object
fallback bound, pose/geometry freshness budget, object/background ownership
contract, target CPU/GPU platform and license-pinned deployment artifacts. The
survey supports a small integration pilot; it does not establish a certified
containment bound or approve real-robot deployment.
