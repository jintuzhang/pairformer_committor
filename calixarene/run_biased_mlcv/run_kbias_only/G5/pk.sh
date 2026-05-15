#!/bin/sh

reweight.py --colvar colvar --cv V2,cyl.z --outfile pk.dat --sigma 0.03,0.03 --temp 300 --bias no --min 0,0.3 --max 3,1.8
python -c "
import plumed
data = plumed.read_as_pandas('pk.dat')
data['pk'] = np.exp(-['file.free'] / 2.49)
plumed(data, 'pk.dat')
"
