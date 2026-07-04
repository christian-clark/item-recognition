# Item Recognition and Cued Recall Evaluations of Transformers

Cued recall evaluation (Experiment 1):

```
python3 scripts/cued_recall.py inputs/names.txt inputs/cities.txt gpt2 --seed 42 --length 20 > raw_output.csv
python3 scripts/surprisal_reduction_cr.py < raw_output.csv > surp_reduction.csv`
```

Item recognition evaluation (Experiment 2):

```
python3 scripts/item_recognition.py inputs/names.txt gpt2 --seed 42 --length 20 > raw_output.csv
python3 scripts/calculate_surprisal_reduction.py < raw_output.csv > surp_reduction.csv
```

`item_recognition_var{1,2,3}.py` contain the variant prompts evaluated in Appendix B.
