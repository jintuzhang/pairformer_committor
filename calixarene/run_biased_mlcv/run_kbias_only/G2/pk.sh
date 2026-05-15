#!/bin/sh

reweight.py --colvar colvar --cv V2,cyl.z --outfile pk.dat --sigma 0.03,0.03 --temp 300 --bias no --min 0,0.3 --max 3,1.8
python -c "
import plumed
import numpy as np
data = plumed.read_as_pandas('pk.dat')
data['pk'] = np.exp(-data['file.free'] / 2.49)
del(data['file.free'])
plumed.write_pandas(data, 'pk.dat')
"
