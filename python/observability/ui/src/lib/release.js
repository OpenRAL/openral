// The Release e-stop control as a pure state machine, so its safety behaviour is tested
// without a DOM: pressable only once armed (RELEASE_ARM_MS after it appears), confirmed on
// the page (a second press while asking), one request at a time, and a failure shows and
// re-enables it. The component (ActionBar.jsx ReleaseButton) only feeds it events and
// sends the request when `send` is true.

export const RELEASE_IDLE = Object.freeze({ armed: false, asking: false, busy: false, error: "" });

// (state, event) -> [next state, send the estop_reset request now]
// events: {type: "arm"} | {type: "press"} | {type: "timeout"} | {type: "result", ok, text}
export function releaseStep(state, event) {
  switch (event?.type) {
    case "arm":
      return [{ ...state, armed: true }, false];
    case "press":
      if (!state.armed || state.busy) return [state, false]; // disabled: nothing happens
      if (!state.asking) return [{ ...state, asking: true, error: "" }, false];
      return [{ ...state, asking: false, busy: true }, true];
    case "timeout": // the confirm window closed without a second press: nothing sent
      return [{ ...state, asking: false }, false];
    case "result":
      return [{ ...state, busy: false, error: event.ok ? "" : String(event.text || "failed") }, false];
    default:
      return [state, false];
  }
}

// What the button shows for `state`.
export function releaseView(state) {
  return {
    disabled: !state.armed || state.busy,
    label: state.asking ? "Confirm release" : "Release e-stop",
    asking: state.asking,
    error: state.error,
  };
}
