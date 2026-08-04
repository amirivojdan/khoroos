# Interpreting results

Khoroos reports observations and indicators, not clinical or welfare diagnoses. Its outputs are
most useful when interpreted alongside flock age, breed, housing, camera placement, recording
time, and environmental measurements.

## Behaviour groups

| Group | Behaviours |
| --- | --- |
| Comfort | preening, dust bathing, wing flapping, stretching, body shaking |
| Locomotion | walking, running |
| Inactive | resting, standing |
| Feeding and drinking | feeding, drinking |
| Foraging | litter pecking, litter scratching |
| Other | head scratching, pooping |

## Read quality signals first

Before interpreting an indicator, check:

1. **Uncertain share.** A large uncertain share can mean the footage differs from the model's
   training data or that birds are difficult to see.
2. **Observation time.** Every budget is based on bird-seconds. Short recordings can produce
   unstable shares and do not trigger alerts below the configured minimum.
3. **Warnings.** High clip rejection, no tracks, or small and occluded birds are reported in the
   result warnings.
4. **Per-class reliability.** Rare actions can be less reliable than common behaviours. Alerts
   are suppressed when reported class reliability is below the configured gate.
5. **Visual agreement.** Review several representative predictions against the source footage.

## Time budgets

A time budget is the share of classified bird-time assigned to each behaviour. Keep uncertain
time visible: reporting only the confident subset can make weak footage appear more conclusive
than it is.

## Welfare indicators and alerts

Indicators aggregate related behaviours, such as locomotion or comfort activity. An alert means
an indicator crossed a configured threshold after observation-time and reliability checks. It
does not identify a cause. Compare runs captured under consistent conditions and investigate
changes using additional observations.

## Track-level results

Per-bird output is approximate because identities may switch during occlusion. Use it to locate
examples and explore patterns. Prefer flock-level measures for formal reporting unless identity
quality has been independently validated for the recording setup.
