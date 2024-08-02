import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


class SeqToSeqDataLoader:
    def __init__(self, dataloader, sequence_length):
        self.dataloader = dataloader
        self.sequence_length = sequence_length
        self.data = self._load_data()
        self.segments = self._create_segments()
        self.current_segment_index = -1
        self.current_index_in_segment = 0
        self.new_segment_flag = True

    def _load_data(self):
        data = []
        for x, y in self.dataloader:
            data.append((x, y))
        return data

    def _create_segments(self):
        segments = [
            self.data[i : i + self.sequence_length]
            for i in range(0, len(self.data), self.sequence_length)
        ]
        segments = [seg for seg in segments if len(seg) == self.sequence_length]
        random.shuffle(segments)
        return segments

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.current_segment_index == -1
            or self.current_index_in_segment >= self.sequence_length
        ):
            self.current_segment_index += 1
            self.current_index_in_segment = 0
            if self.current_segment_index >= len(self.segments):
                self.current_segment_index = 0
                random.shuffle(self.segments)
            self.new_segment_flag = True
        else:
            self.new_segment_flag = False

        segment = self.segments[self.current_segment_index]
        sample = segment[self.current_index_in_segment]
        self.current_index_in_segment += 1

        return sample[0], sample[1], self.new_segment_flag

    def __len__(self):
        return len(self.data)
