#!/bin/bash

s_vals=("sqrt")

N_vals=(1000)

p_vals=("0.1" "0.25" "0.50" "0.75")

d_vals=("1" "2" "12")

for s in ${s_vals[@]}; do
    for N in ${N_vals[@]}; do
        for p in ${p_vals[@]}; do
            echo "$s  $N  0.00  0"
            python failures_LR.py "$s"  "$N"  "0.00"  "0"
            for d in ${d_vals[@]}; do
                echo "$s $N $p $d"
                python failures_LR.py "$s"  "$N"  "$p"  "$d"
            done
        done
    done
done
            
