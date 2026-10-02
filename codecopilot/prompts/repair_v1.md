---
id: repair_v1
description: One-shot citation repair - feed the checker's findings back and ask for a corrected answer
---
[system]
(appended as a follow-up turn to the answer conversation)
[user]
An automatic checker compared your answer with the code excerpts and found these citation problems:

{problems}

Rewrite the whole answer so that:
- every claim cites a narrow line range (using the line numbers shown) that actually contains the names it mentions,
- any claim you cannot support with the excerpts is removed.
Output only the corrected answer.
