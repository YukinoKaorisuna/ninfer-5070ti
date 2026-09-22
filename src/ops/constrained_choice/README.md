# constrained_choice

M1-A implementation target.

This op gathers a small finite set of token logits from the model's existing
full-vocabulary BF16 output and computes a softmax only across those choices.

It intentionally does not modify the LM head or generation runtime yet.

Files to add next:

- constrained_choice.cu
- launcher wiring
- public wrapper validation
- numerical test
- microbenchmark

Initial supported geometry:

- B: 1..8
- K: 2..16
