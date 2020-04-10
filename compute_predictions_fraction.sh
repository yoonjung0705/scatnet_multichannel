#!/bin/sh
IDX_STARTS=$(seq 0 100 900)
for IDX_START in ${IDX_STARTS}
do
    echo ${IDX_START}
    cat "compute_predictions_fraction_template.py" | sed \
        "s/<IDX_START>/${IDX_START}/g" > compute_predictions_fraction.py
    python compute_predictions_fraction.py
done
