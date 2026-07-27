# Writeup

## Approach and Design Decisions

I built a two-stage LLM support triage pipeline. The classification stage receives only the relevant ticket fields (`ticket_id`, `subject`, `body`, and `metadata`) plus the classification prompt, which includes the relevant taxonomy and no-draft rules. It returns the structured triage fields: category, urgency, `should_draft`, `no_draft_reason`, `draft_response`, and confidence.

I separated classification from drafting because they are different tasks. Classification is a constrained decision over preset labels and safety rules; drafting is open-ended text generation. Keeping them separate avoids mixing draft-writing instructions into the classifier and makes it possible to skip drafting entirely for sensitive tickets. The draft stage only runs when `should_draft == true`, and it receives both the original ticket JSON and the classification result.

The pipeline is provider-agnostic. Model choices live in `src/config.py` as named configs, and all provider-specific OpenAI/Anthropic logic is isolated in `src/llm.py`. Each results directory includes a copy of `config.py` so runs are reproducible. I selected the `claude-sonnet` config for the final training run because it had the best observed training performance while still balancing speed, cost, and accuracy.

## Evaluation Results

On the labeled training set (`30` tickets), the final `claude-sonnet` run produced:

- Category accuracy: `100.00%` (`30/30`)
- Urgency accuracy: `86.67%` (`26/30`)
- Should-draft accuracy: `96.67%` (`29/30`)
- Sensitive tickets: `9`
- False drafts on sensitive tickets: `0`
- False draft rate: `0.00%`
- Draftable tickets: `21`
- False no-drafts: `1`
- False no-draft rate: `4.76%`
- Average confidence: `0.835`
- Average confidence on correct predictions: `0.852`
- Average confidence on incorrect predictions: `0.754`

On the unlabeled eval set, I compared `claude-sonnet` against `gpt-4.1-mini` using agreement metrics rather than treating either model as ground truth:

- Compared tickets: `15`
- Category agreement: `100.00%` (`15/15`)
- Urgency agreement: `60.00%` (`9/15`)
- Should-draft agreement: `100.00%` (`15/15`)
- Draft/no-draft disagreement count: `0`
- Tickets with any core-field disagreement: `6`
- Average confidence, `claude-sonnet`: `0.855`
- Average confidence, `gpt-4.1-mini`: `0.949`

The eval-set disagreements were all urgency disagreements, not category or drafting-safety disagreements.

## Failure Modes

The main failure mode is urgency calibration. On the training set, the model got all categories correct and had zero false drafts on sensitive tickets, but urgency was wrong on four tickets. The mistakes were mostly adjacent severity errors: `low` vs `medium`, or `high` vs `medium`. This suggests the classifier understands the issue type but is less consistent about operational priority.

The one should-draft error was conservative: a duplicate ACH deposit ticket was marked no-draft even though the ground truth allowed drafting. This is preferable to the opposite failure mode in a regulated support context, but it could increase human review load.

The unlabeled eval agreement results reinforce the same pattern. The two configs agreed on every category and should-draft decision, but disagreed on urgency for six of fifteen tickets. Urgency definitions likely need sharper examples or a small calibration layer.

## Next Steps

With another week, I would add a small targeted calibration set focused on urgency boundaries, especially `low` vs `medium` and `medium` vs `high`. I would also add few-shot examples for common ambiguous cases like deposit limits, small payment anomalies, address changes, and tax-history requests.

I would add automated regression tests that run the labeled training set and fail if false drafts on sensitive tickets become nonzero. I would also expand the agreement report to include per-ticket side-by-side model outputs and confidence deltas so review time is spent on the riskiest disagreements first.
