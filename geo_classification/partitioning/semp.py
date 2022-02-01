import json
import logging
from typing import List
from pathlib import Path
import numpy as np

from . import utils as putils


class SemPHierarchy:
    def __init__(
        self, partitionings: List[putils.BasePartitioning], parents_dict_path: Path
    ):
        """
         M[class_index_coarse, 0] -> child in class_index_middle
         M[class_index_middle, 1] -> child in class_index_fine

         M = array([[  438,     0,     0],
        [  685,  1561,     1],
        [  580,  2928,     2],
        ...,
        [    0,   681, 13503],
        [    0,   422, 13504],
        [    0,   275, 13505]], dtype=int32)

        """
        self.partitionings = partitionings
        with open(parents_dict_path) as fr:
            self.parents_dict = json.load(fr)
        logging.debug(self.parents_dict.keys())

        self.M = self.__build_hierarchy()

    def __build_hierarchy(self):

        cell_hierarchy = np.zeros(
            (len(self.partitionings[-1]), len(self.partitionings)), dtype=np.int32
        )

        for pidx in reversed(range(len(self.partitionings))):
            if pidx == 0:
                break
            logging.debug(f"### {pidx} ###")
            logging.debug(len(self.partitionings[pidx]))
            if pidx == len(self.partitionings) - 1:
                for i in range(len(self.partitionings[pidx])):
                    cell_hierarchy[i, pidx] = i
            for loc_scr, loc_par in self.parents_dict[
                f"m_{len(self.partitionings[pidx])}"
            ].items():
                src = self.partitionings[pidx].label2index(loc_scr)
                par = self.partitionings[pidx - 1].label2index(loc_par)
                cell_hierarchy[src, pidx - 1] = par

        return cell_hierarchy
