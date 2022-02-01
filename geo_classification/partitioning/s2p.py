import logging
from typing import List

import numpy as np
import s2sphere as s2

from . import utils as putils


class S2Hierarchy:
    def __init__(self, partitionings: List[putils.BasePartitioning]):

        """
        Provide a matrix of class indices where each class of the finest partitioning will be assigned
        to the next coarser scales.

        Resulting index matrix M has shape: max(classes) * |partitionings| and is ordered from coarse to fine
        """
        self.partitionings = partitionings

        putils.print_partitioning_stats(self.partitionings)

        self.M = self.__build_hierarchy()

    def __build_hierarchy(self):
        def _hextobin(hexval):
            thelen = len(hexval) * 4
            binval = bin(int(hexval, 16))[2:]
            while (len(binval)) < thelen:
                binval = "0" + binval

            binval = binval.rstrip("0")
            return binval

        def _create_cell(lat, lng, level):
            p1 = s2.LatLng.from_degrees(lat, lng)
            cell = s2.Cell.from_lat_lng(p1)
            cell_parent = cell.id().parent(level)
            hexid = cell_parent.to_token()
            return hexid

        cell_hierarchy = []

        finest_partitioning = self.partitionings[-1]
        logging.info("Create hierarchy from partitionings...")
        if len(self.partitionings) > 1:
            # loop through finest partitioning
            for c in range(len(finest_partitioning)):
                cell_bin = _hextobin(self.partitionings[-1].get_class_label(c))
                level = int(len(cell_bin[3:-1]) / 2)
                parents = []

                # get parent cells
                for l in reversed(range(2, level + 1)):
                    lat, lng = finest_partitioning.get_lat_lng(c)
                    hexid_parent = _create_cell(lat, lng, l)
                    # to coarsest partitioning
                    for p in reversed(range(len(self.partitionings))):
                        if self.partitionings[p].contains(hexid_parent):
                            parents.append(
                                self.partitionings[p].label2index(hexid_parent)
                            )

                    if len(parents) == len(self.partitionings):
                        break

                cell_hierarchy.append(parents[::-1])
        logging.info("Finished.")
        M = np.array(cell_hierarchy, dtype=np.int32)
        assert max([len(p) for p in self.partitionings]) == M.shape[0]
        assert len(self.partitionings) == M.shape[1]
        logging.debug(M)
        logging.info(f"M={M.shape}")
        return M
