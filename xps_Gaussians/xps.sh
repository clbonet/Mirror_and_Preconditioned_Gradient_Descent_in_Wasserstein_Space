#!/bin/bash

target=non_diag
methods=(nem fb pfb klm pklm svgd wgd pwgd)

for method in ${methods[*]}
do
	echo $method
	python xp_gaussian.py --method $method --target $target --pbar
done

