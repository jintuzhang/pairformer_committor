#/bin/sh

../../../../reweight.py --colvar colvar --cv V2,cyl.z --outfile fes.2d.dat --sigma 0.04,0.04 --temp 300 --min 0,0.3 --max 3,1.8 --skiprows 2000 --bias w_1.bias,w_cap.bias,opes.bias
