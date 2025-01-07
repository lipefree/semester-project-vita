from dataclasses import dataclass
from typing import List
import numpy as np


class ClassificationLogger():

    def __init__(self, writer):
        self.global_step: int = 0
        self.global_osm_selected:int = 0
        self.global_sat_selected:int = 0 
        self.global_osm_true:int = 0
        self.global_sat_true:int = 0 

        self.update_step:int = 0
        self.nbr_sample:int = 0
        self.nbr_correct_sample:int = 0
        self.sat_mean: List = []
        self.osm_mean: List = []

        self.window_label: List = []
        self.window_pred: List = []

        self.window_sat_selected: List = []

        self.distance: List = []
        self.window_size = 2000
        self.batch_step = 0
        self.logging_time = 50

        self.writer = writer

    # external
    def log_step(self, chosen_cpu, gt_choices_cpu, osm_dist, sat_dist, bs):
        self.log_dists(osm_dist, sat_dist)
        self.log_osm_sat_count(chosen_cpu, gt_choices_cpu)
        self.log_label_preds(chosen_cpu, gt_choices_cpu)

        self.batch_step += bs

        if self.batch_step % self.logging_time:
            self.write_logs()

    def log_dists(self, osm_dist, sat_dist):
        self.sat_mean.append(np.mean(sat_dist))
        self.osm_mean.append(np.mean(osm_dist))

    def count_osm_sat(self, list):
        num_osm = np.sum(list)
        return num_osm, len(list) - num_osm

    def log_osm_sat_count(self, chosen_cpu, gt_choices_cpu):
        num_osm, num_sat = self.count_osm_sat(chosen_cpu)
        self.osm_true += num_osm
        self.sat_true += num_sat
        num_osm, num_sat = self.count_osm_sat(gt_choices_cpu)
        self.osm_true += num_osm
        self.sat_true += num_sat

    def log_label_preds(self, chosen_cpu, gt_choice_cpu):
        self.window_label.extends(np.array(gt_choice_cpu).flatten())
        self.window_pred.extends(np.array(chosen_cpu).flatten())

    def write_logs(self):
        self.writer.add_scalar("train/dist_diff", (self.sat_dist - self.osm_dist) ,self.global_step)
        self.writer.add_scalar("train/dist_osm", np.abs(self.osm_dist) ,self.global_step)
        self.writer.add_scalar("train/dist_sat", np.abs(self.sat_dist) ,self.global_step)
        self.writer.add_scalar("train/dist_sat_mean", np.mean(self.sat_mean),self.global_step)
        self.writer.add_scalar("train/dist_osm_mean", np.mean(self.osm_mean),self.global_step)

        

# internally every 200 steps, it logs

