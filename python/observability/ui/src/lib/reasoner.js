// Why a prompt cannot reach the reasoner, from its last lifecycle transition.
// `lifecycle` is the snapshot's topics.lifecycle ({node: {transition, result}}),
// filled from the `deploy.bringup` spans every lifecycle callback emits. No entry
// means we have not heard (the dashboard may have started after the reasoner did),
// so the bar stays usable rather than guessing.
const STOPPING = new Set(["on_deactivate", "on_cleanup", "on_shutdown"]);

export function reasonerDown(lifecycle) {
    const e = lifecycle?.openral_reasoner;
    if (!e || typeof e !== "object") return null;
    const step =
        typeof e.transition === "string" ? e.transition : "a transition";
    if (e.result !== "SUCCESS")
        return `The reasoner failed to start (${step} failed; see Events), so a prompt would go nowhere.`;
    if (STOPPING.has(e.transition))
        return "The reasoner is stopped, so a prompt would go nowhere.";
    return null;
}
