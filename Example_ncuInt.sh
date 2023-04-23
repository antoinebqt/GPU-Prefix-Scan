#!/usr/bin/bash
export TMPDIR="${HOME}/tmp"

mkdir $TMPDIR 2> /dev/null
echo "TMPDIR set to $TMPDIR"
if [ "$#" -le 1 ]; 
    then echo "illegal number of parameters"; echo "$0 <program>"
    exit -1
fi

echo "Collecting Double Precision metrics"
ncu --target-processes all  --metrics \
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum"    $@