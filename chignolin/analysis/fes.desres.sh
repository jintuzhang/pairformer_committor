#!/bin/sh

python append.py

../../reweight/reweight.py \
    --colvar colvar.desres \
    --cv d_asp3n_gly7o,d_asp3n_thr8o \
    --outfile fes.desres.dat \
    --sigma 0.02,0.02 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340

../../reweight/reweight.py \
    --colvar colvar.desres \
    --cv d_asp3n_gly7o,d_ee \
    --outfile fes.desres.1.dat \
    --sigma 0.02,0.02 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340

../../reweight/reweight.py \
    --colvar colvar.desres \
    --cv d_asp3od_thr6n,d_asp3n_thr8o \
    --outfile fes.desres.2.dat \
    --sigma 0.02,0.02 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340

../../reweight/reweight.py \
    --colvar colvar.desres \
    --cv d_asp3n_gly7o,d_asp3od_thr6n \
    --outfile fes.desres.3.dat \
    --sigma 0.02,0.02 \
    --bin 200,200 \
    --min 0.2,0.2 \
    --max 1.8,1.8 \
    --temp 340
