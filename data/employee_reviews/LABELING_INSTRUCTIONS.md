# Labeling rubric — Employee Reviews work-location classification

Each review must be classified into exactly one of three classes:

## remote
The review states the employee works remotely, from home, or that the company
offers/requires remote work. Includes hybrid arrangements where the review
emphasises the remote component.

Example: "Loved being able to work from home full time. Saved me 2 hours a day in commute."

## not_remote
The review states the employee works on-site, in an office, has to commute, or
that the company requires in-person attendance. Includes return-to-office
mandates and hybrid where the review emphasises the on-site component.

Example: "The office is downtown and parking is impossible. We're expected in 5 days a week."

## not_mentioned
The review does not discuss work location at all, or mentions location only
incidentally without describing the arrangement.

Example: "Great managers and good benefits. The team really supports each other."

## Edge cases — when in doubt
- "Flexible hours" alone is NOT a location signal → not_mentioned unless explicitly remote-related.
- City names alone with no further context → lean not_mentioned.
- "Used to work remote, now in office" → not_remote (current state matters).
- Mixed hybrid with no emphasis → lean toward what's mentioned more.

## Process
1. Open a new Claude.ai chat.
2. Paste this entire file as context.
3. Upload one batch CSV.
4. Ask: "Fill the `label` column with exactly one of: remote, not_remote, not_mentioned. Output as CSV with the same columns and same row order. No commentary. Be conservative — when unsure, use not_mentioned."
5. Save the returned CSV as `batch_N_labeled.csv` in `data/employee_reviews/to_label/`.
6. Read through and fix obvious errors before merging.
