import json
import os

import lightning as L

from litdata import (CombinedStreamingDataset, StreamingDataLoader,
                     StreamingDataset, TokensLoader, train_test_split)

_MAX_VAL_RATIO = 0.1


class TextDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        max_seq_len,
        num_val_samples,
        train_batch_size,
        val_batch_size,
        *args,
        num_proc=1,
        shuffle=True,
        **kwargs,
    ):

        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self._train_dataloader = None
        self._val_dataloader = None

        self.max_seq_len = max_seq_len

        self.val_size = num_val_samples
        self.shuffle = shuffle

        self.num_proc = num_proc

        self.additional_args = args
        self.additional_kwargs = kwargs

        self.config_description = None

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            if self.data_path.endswith(".json"):
                mixture_dict = json.load(open(self.data_path, "r"))
                self.config_description = mixture_dict

                dataset_paths = list(mixture_dict.keys())
                dataset_weights = list(mixture_dict.values())

                train_val_datasets = [
                    StreamingDataset(
                        input_dir=dataset_path,
                        item_loader=TokensLoader(block_size=self.max_seq_len + 1),
                        shuffle=self.shuffle,
                        drop_last=True,
                    )
                    for dataset_path in dataset_paths
                ]

                for i_ds, ds in enumerate(train_val_datasets):
                    val_ratio = self.val_size / (len(ds) * len(train_val_datasets))
                    val_ratio = min(val_ratio, _MAX_VAL_RATIO)
                    train_val_datasets[i_ds] = train_test_split(
                        ds, [1.0 - val_ratio, val_ratio]
                    )

                self.train_dataset = CombinedStreamingDataset(
                    datasets=[el[0] for el in train_val_datasets],
                    weights=dataset_weights,
                    iterate_over_all=False,
                )

                self.val_dataset = CombinedStreamingDataset(
                    datasets=[el[1] for el in train_val_datasets],
                    weights=dataset_weights,
                    iterate_over_all=False,
                )
            else:
                self.config_description = {self.data_path: 1}

                path_subdirs = os.listdir(self.data_path)
                if "index.json" not in path_subdirs:
                    data_paths = [
                        os.path.join(self.data_path, subdir) for subdir in path_subdirs
                    ]
                    try:
                        subdir_datasets = [
                            StreamingDataset(
                                input_dir=sub_data_path,
                                item_loader=TokensLoader(
                                    block_size=self.max_seq_len + 1
                                ),
                                shuffle=self.shuffle,
                                drop_last=True,
                            )
                            for sub_data_path in data_paths
                        ]
                    except ValueError:
                        raise ValueError(
                            f"{self.data_path} should either contain litdata processed files or subfolders containing litdata processed files."
                        )

                    val_ratio = self.val_size / len(ds)
                    val_ratio = min(val_ratio, _MAX_VAL_RATIO)

                    subdir_datasets_split = [
                        train_test_split(ds, [1.0 - val_ratio, val_ratio])
                        for ds in subdir_datasets
                    ]

                    self.train_dataset = CombinedStreamingDataset(
                        datasets=[ds[0] for ds in subdir_datasets_split],
                        iterate_over_all=False,
                    )
                    self.val_dataset = CombinedStreamingDataset(
                        datasets=[ds[1] for ds in subdir_datasets_split],
                        iterate_over_all=False,
                    )

                else:
                    dataset = StreamingDataset(
                        input_dir=self.data_path,
                        item_loader=TokensLoader(block_size=self.max_seq_len + 1),
                        shuffle=self.shuffle,
                        drop_last=True,
                    )
                    val_ratio = self.val_size / len(dataset)
                    val_ratio = min(val_ratio, _MAX_VAL_RATIO)

                    self.train_dataset, self.val_dataset = train_test_split(
                        dataset, [1.0 - val_ratio, val_ratio]
                    )

    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = StreamingDataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_proc,
                pin_memory=False,
            )
        return self._train_dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = StreamingDataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_proc,
                pin_memory=False,
                persistent_workers=True,
            )
        return self._val_dataloader

    def state_dict(self):
        # track whatever you want here
        state = {"train_dl_state_dict": self._train_dataloader.state_dict()}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.train_dataloader().load_state_dict(state_dict["train_dl_state_dict"])
