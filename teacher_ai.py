
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

MAX_TOOL_ROUNDS = 3

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_bearing_spectrum",
            "description": (
                "Measure BPFO and BPFI harmonic peaks "
                "from a recorded bearing vibration signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bearing_id": {
                        "type": "string"
                    },
                    "recording": {
                        "type": "integer"
                    }
                },
                "required": [
                    "bearing_id",
                    "recording"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_fault_evidence",
            "description": (
                "Compare measured BPFO and BPFI "
                "harmonic amplitudes and their ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bearing_id": {
                        "type": "string"
                    },
                    "recording": {
                        "type": "integer"
                    }
                },
                "required": [
                    "bearing_id",
                    "recording"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_signal_quality",
            "description": (
                "Inspect recorded vibration RMS, "
                "peak amplitude, crest factor, "
                "and missing values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bearing_id": {
                        "type": "string"
                    },
                    "recording": {
                        "type": "integer"
                    }
                },
                "required": [
                    "bearing_id",
                    "recording"
                ],
                "additionalProperties": False
            }
        }
    }
]



SYSTEM_PROMPT = """
You are NVIDIA Nemotron acting as an evidence-grounded
diagnostic investigator inside EdgeMentor, an industrial
bearing-monitoring prototype.

Your role is to independently investigate a bearing case
using approved diagnostic tools. The edge classifier may
be wrong, including when its confidence is high.

You have three diagnostic tools:
1. inspect_bearing_spectrum
2. compare_fault_evidence
3. inspect_signal_quality

INVESTIGATION RULES:

1. EVIDENCE AND UNCERTAINTY
- Request diagnostic tools when their measurements can help.
- Use only the bearing and recording specified in the case.
- Use actual tool outputs. Never invent measurements.
- Treat the edge prediction, confidence, anomaly flag, and
  supervisor decision as fallible inputs, not ground truth.
- High classifier confidence does not establish correctness.
- Do not use the classifier's prediction as independent
  evidence supporting that same prediction.
- Distinguish measured facts, interpretations, and unknowns.

2. NO UNSUPPORTED BASELINES
- Never call a measurement normal, abnormal, high, low,
  typical, or unusual without a relevant reference or
  validated threshold in the available evidence.
- Do not invent healthy ranges, baseline values, thresholds,
  or diagnostic probabilities.
- Do not infer that a bearing is healthy merely because
  the waveform has no missing or non-finite samples.
- A high crest factor alone does not prove a bearing fault.
- BPFO/BPFI peaks and amplitude ratios are descriptive
  evidence, not proof of a specific defect.
- Do not claim that peaks are higher than healthy baselines
  unless the tools provide a valid comparison.

3. DIAGNOSTIC REASONING
- Consider multiple plausible explanations.
- Explain which observations support each hypothesis and
  which observations are inconclusive or contradictory.
- Do not force a fault diagnosis when the evidence is
  insufficient.
- Do not assume that a spectral peak is caused by a defect
  solely because it lies near a theoretical fault frequency.
- If the tools do not distinguish the hypotheses, state that
  explicitly and recommend additional measurements.

4. SAFETY AND AUTHORITY
- Never claim a fault has been confirmed unless the available
  evidence genuinely establishes it; this prototype's current
  tools alone should not be treated as physical confirmation.
- Never silently change the edge classifier's prediction
  or the local supervisor's decision.
- Never recommend automatic shutdown or repair based only
  on this investigation.
- Recommend qualified human review when uncertainty remains.

REPORT FORMAT:

## Investigation Summary
Identify the bearing, recording, original edge prediction,
confidence, and supervisor decision. Explain that this is
an investigation, not an automatic correction.

## Measured Evidence
Report relevant values and units exactly as returned.
Identify which tool supplied each measurement.
Do not add unsupported reference ranges or comparisons.

## Competing Hypotheses
Present plausible explanations and the evidence for and
against each. Clearly mark unresolved possibilities.

## Recommended Next Checks
Suggest practical follow-up measurements or inspections
that could distinguish the competing explanations.
Do not imply that unavailable checks have already occurred.

## Final Assessment
Summarize what the evidence supports and what remains
unknown. Recommend further investigation when warranted.
Do not claim certainty or automatically change the edge
decision.

Keep the report specific, concise, and faithful to the
measurements. Never sacrifice accuracy to sound decisive.
"""



def _get_client():
    api_key = os.getenv("NEBIUS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing."
        )

    return OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=api_key,
        timeout=60.0,
        max_retries=0
    )


def investigate_bearing(
    bearing,
    recording,
    prediction,
    confidence,
    low_confidence,
    anomaly_flag,
    decision="ESCALATE"
):
    """
    Execute a bounded Nemotron diagnostic investigation.

    Returns:
        {
            "report": str,
            "tool_trace": list,
            "tool_calls_count": int,
            "status": str
        }
    """

    if decision not in ("ACCEPT", "ESCALATE"):
        return {
            "report": (
                "Investigation unavailable: "
                "this case is not eligible for diagnostic analysis. "
                "Invalid sensor readings must be corrected first."
            ),
            "tool_trace": [],
            "tool_calls_count": 0,
            "status": "BLOCKED"
        }

    # Import here to avoid the circular dependency:
    # edgementor -> teacher_ai -> diagnostic_tools -> edgementor
    from diagnostic_tools import DIAGNOSTIC_TOOLS

    try:
        client = _get_client()

        case = {
            "bearing_id": bearing,
            "recording": recording,
            "prediction": prediction,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "anomaly_flag": anomaly_flag,
            "supervisor_decision": decision
        }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Investigate this escalated bearing case. "
                    "Request measured evidence before forming "
                    "your assessment.\n\n"
                    + json.dumps(case, indent=2)
                )
            }
        ]

        trace = []
        tool_calls_count = 0

        for round_number in range(MAX_TOOL_ROUNDS):

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_completion_tokens=2500
            )

            message = response.choices[0].message

            messages.append(
                message.model_dump(
                    exclude_none=True
                )
            )

            if not message.tool_calls:
                return {
                    "report": (
                        message.content
                        or "No investigation report returned."
                    ),
                    "tool_trace": trace,
                    "tool_calls_count": tool_calls_count,
                    "status": "COMPLETED"
                }

            for call in message.tool_calls:

                name = call.function.name

                tool_calls_count += 1

                try:
                    arguments = json.loads(
                        call.function.arguments
                    )

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be an object."
                        )

                    # Enforce the selected case.
                    # Nemotron cannot switch bearings
                    # or inspect unrelated recordings.
                    if (
                        arguments.get("bearing_id") != bearing
                        or arguments.get("recording") != recording
                    ):
                        raise ValueError(
                            "Tool request does not match "
                            "the selected bearing and recording."
                        )

                    if name not in DIAGNOSTIC_TOOLS:
                        raise ValueError(
                            "Unapproved diagnostic tool."
                        )

                    result = DIAGNOSTIC_TOOLS[name](
                        bearing_id=bearing,
                        recording=recording
                    )

                    trace.append({
                        "round": round_number + 1,
                        "tool": name,
                        "result": result
                    })

                except Exception as error:

                    result = {
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        )
                    }

                    trace.append({
                        "round": round_number + 1,
                        "tool": name,
                        "result": result
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

        # Tool budget exhausted. Force a final
        # answer from the evidence already collected.
        messages.append({
            "role": "user",
            "content": (
                "The diagnostic tool budget is exhausted. "
                "Produce your final evidence-grounded report "
                "using only the available measurements. "
                "Clearly state unresolved uncertainty."
            )
        })

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tool_choice="none",
            max_completion_tokens=2500
        )

        report = (
            response.choices[0].message.content
            or "No final report returned."
        )

        return {
            "report": report,
            "tool_trace": trace,
            "tool_calls_count": tool_calls_count,
            "status": "COMPLETED"
        }

    except Exception as error:
        return {
            "report": (
                "Investigation unavailable: "
                f"{type(error).__name__}: {error}"
            ),
            "tool_trace": (
                trace if "trace" in locals() else []
            ),
            "tool_calls_count": (
                tool_calls_count
                if "tool_calls_count" in locals()
                else 0
            ),
            "status": "ERROR"
        }


# Preserve compatibility with existing dashboard
# and edgementor.py calls.
def ask_teacher(
    prediction,
    confidence,
    low_confidence,
    anomaly_flag
):
    """
    Legacy advisory interface.

    The full tool-using investigator requires
    bearing ID and recording number.
    """

    return (
        "The Nemotron Diagnostic Investigator is available. "
        "Use investigate_bearing() with the bearing ID "
        "and recording number to run an evidence-grounded "
        "investigation."
    )


if __name__ == "__main__":

    result = investigate_bearing(
        bearing="KA05",
        recording=1,
        prediction="Outer Damage",
        confidence=0.48,
        low_confidence=True,
        anomaly_flag=False,
        decision="ESCALATE"
    )

    print("\n===== INVESTIGATION STATUS =====")
    print(result["status"])

    print("\n===== TOOLS EXECUTED =====")
    for entry in result["tool_trace"]:
        print(
            f"Round {entry['round']}: "
            f"{entry['tool']}"
        )

    print("\n===== FINAL REPORT =====")
    print(result["report"])
