#!/bin/bash
for crate in libertem_dectris libertem_asi_tpx3 libertem_asi_mpx3 libertem_qd_mpx; do
  (cd $crate && cargo bundle-licenses --format yaml --output THIRDPARTY.yml);
done
