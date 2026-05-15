#!/bin/sh

python append.py

../../../reweight/reweight.py \
    --colvar colvar \
    --cv d_asp3n_gly7o,d_asp3n_thr8o \
    --outfile pk_1.dat \
    --sigma 0.015,0.015 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340 \
    --bias no \
    --skiprow 5000

../../../reweight/reweight.py \
    --colvar colvar \
    --cv d_ee,d_asp3n_thr8o \
    --outfile pk_3.dat \
    --sigma 0.015,0.015 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340 \
    --bias no \
    --skiprow 5000

../../../reweight/reweight.py \
    --colvar colvar \
    --cv d_asp3od_thr6n,d_asp3n_thr8o \
    --outfile pk_5.dat \
    --sigma 0.015,0.015 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340 \
    --bias no \
    --skiprow 5000

../../../reweight/reweight.py \
    --colvar colvar \
    --cv d_asp3n_gly7o,d_asp3od_thr6n \
    --outfile pk_7.dat \
    --sigma 0.015,0.015 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340 \
    --bias no \
    --skiprow 5000

../../../reweight/reweight.py \
    --colvar colvar \
    --cv d_asp3n_gly7o,d_ee \
    --outfile pk_9.dat \
    --sigma 0.015,0.015 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340 \
    --bias no \
    --skiprow 5000
