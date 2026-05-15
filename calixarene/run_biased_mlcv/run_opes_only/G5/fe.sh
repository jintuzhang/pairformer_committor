#/bin/sh

../../../../reweight/reweight.py --colvar colvar --cv cyl.z --outfile fes.dat --sigma 0.01 --temp 300 --bin 200 --skiprow 2000 --bias w_1.bias,w_cap.bias,opes.bias
python ./funnel.py
