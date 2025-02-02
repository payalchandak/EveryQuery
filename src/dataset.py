import rootutils
from meds_torch.data.components.pytorch_dataset import PytorchDataset
from mixins import SeedableMixin

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import random

import numpy as np
import omegaconf
import polars as pl
import scipy
import torch


class EveryQueryDataset(PytorchDataset):
    def __init__(self, cfg):
        cfg.do_include_subsequence_indices = True
        super().__init__(cfg)

        self.code_strategies = ["uniform", "frequency"]
        if self.config.code_sampling_strategy not in self.code_strategies:
            raise ValueError(
                f"code_sampling_strategy must be one of {self.code_strategies}."
            )
        self.value_strategies = ["ignore", "random_quantile", "random_normal", "manual"]
        if self.config.default_value_sampling_strategy not in self.value_strategies:
            raise ValueError(
                f"default_value_sampling_strategy must be one of {self.value_strategies}"
            )
        
        df = self._load_data()
        self._metadata_dict = df.to_dicts()
        self._metadata_schema = df.schema

        self._code_options_dict = {}
        self._code_options_schema = None

        obj = self.config.get("codes", None)
        if obj is not None:
            if not isinstance(obj, omegaconf.listconfig.ListConfig):
                raise TypeError(f"codes must be a list, got {type(obj)}")
        self.set_codes(obj)

        for param, expected_type in [
            ("values_ignore", omegaconf.listconfig.ListConfig),
            ("values_random_quantile", omegaconf.listconfig.ListConfig),
            ("values_random_normal", omegaconf.listconfig.ListConfig),
            ("values_manual", omegaconf.dictconfig.DictConfig),
        ]:
            obj = self.config.get(param, None)
            if obj is not None:
                if not isinstance(obj, expected_type):
                    raise TypeError(
                        f"{param} should be {expected_type}, but got {type(obj)}"
                    )
                self.set_values(strategy=param.replace("values_", ""), data=obj)

        # future
        self.future_strategies = ["within_record", "random", "fixed", "categorical"]
        if self.config.min_future < 0:
            raise ValueError("min_query_future must be non-negative.")
        if self.config.min_future > self.config.max_future:
            raise ValueError("min_query_future must not be greater than max_query_future.")
        
        # duration
        if self.config.duration_sampling_strategy not in self.future_strategies:
            raise ValueError(
                f"duration_sampling_strategy must be one of {self.future_strategies}."
            )
        if not self.config.min_future <= self.config.min_duration <= self.config.max_future: 
            raise ValueError(f"min_duration must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        if not self.config.min_future <= self.config.max_duration <= self.config.max_future: 
            raise ValueError(f"max_duration must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        if self.config.duration_sampling_strategy == "fixed":
            if self.config.fixed_duration is None:
                raise ValueError(
                    "fixed_duration must be specified for 'fixed' sampling strategy."
                )
            if not self.config.min_future <= self.config.fixed_duration <= self.config.max_future: 
                raise ValueError(f"fixed_duration must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        if self.config.duration_sampling_strategy == "categorical":
            if self.config.categorical_duration is None:
                raise ValueError(
                    "categorical_duration must be specified for 'categorical' sampling strategy."
                )
            if not isinstance(self.config.categorical_duration, omegaconf.listconfig.ListConfig):
                raise ValueError(
                    f"categorical_duration should be {omegaconf.listconfig.ListConfig}, but got {type(self.config.categorical_duration)}."
                )
            for value in self.config.categorical_duration: 
                if not self.config.min_future <= value<= self.config.max_future: 
                    raise ValueError(f"value {value} in categorical_duration must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        
        # offset
        if self.config.offset_sampling_strategy not in self.future_strategies:
            raise ValueError(
                f"offset_sampling_strategy must be one of {self.future_strategies}."
            )
        if not self.config.min_future <= self.config.min_offset <= self.config.max_future: 
            raise ValueError(f"min_offset must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        if not self.config.min_future <= self.config.max_offset <= self.config.max_future: 
            raise ValueError(f"max_offset must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        if self.config.offset_sampling_strategy == "fixed":
            if self.config.fixed_offset is None:
                raise ValueError(
                    "fixed_offset must be specified for 'fixed' sampling strategy."
                )
            if not self.config.min_future <= self.config.fixed_offset <= self.config.max_future: 
                raise ValueError(f"fixed_offset must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        if self.config.offset_sampling_strategy == "categorical":
            if self.config.categorical_offset is None:
                raise ValueError(
                    "categorical_offset must be specified for 'categorical' sampling strategy."
                )
            if not isinstance(self.config.categorical_offset, omegaconf.listconfig.ListConfig):
                raise ValueError(
                    f"categorical_offset should be {omegaconf.listconfig.ListConfig}, but got {type(self.config.categorical_offset)}."
                )
            for value in self.config.categorical_offset: 
                if not self.config.min_future <= value<= self.config.max_future: 
                    raise ValueError(f"value {value} in categorical_offset must be in future bounds ({self.config.min_future}, {self.config.max_future}).")
        
    @property
    def metadata(self):
        return pl.DataFrame(self._metadata_dict, schema=pl.Schema(self._metadata_schema))  

    @metadata.setter
    def metadata(self, value):
        if isinstance(value, pl.DataFrame):
            self._metadata_dict = value.to_dicts() 
            self._metadata_schema = value.schema
        elif isinstance(value, list):
            self._metadata_dict = value
        else:
            raise TypeError("metadata must be a Polars DataFrame or a list of dictionaries.")

    @property
    def code_options(self):
        return pl.DataFrame(self._code_options_dict, schema=pl.Schema(self._code_options_schema))

    @code_options.setter
    def code_options(self, value):
        if isinstance(value, pl.DataFrame):
            self._code_options_dict = value.to_dicts()
            self._code_options_schema = value.schema
        elif isinstance(value, list):
            self._code_options_dict = value
        else:
            raise TypeError("code_options must be a Polars DataFrame or a list of dictionaries.")

    def _load_data(self):
        return (
            pl.read_parquet(self.config.code_metadata_fp)
            .filter(pl.col("code").is_not_null())
            .with_columns(
                pl.col("values/quantiles").struct.field("values/quantile/0").alias("values/quantile/0"),
                pl.col("values/quantiles").struct.field("values/quantile/0.1").alias("values/quantile/10"),
                pl.col("values/quantiles").struct.field("values/quantile/0.2").alias("values/quantile/20"),
                pl.col("values/quantiles").struct.field("values/quantile/0.3").alias("values/quantile/30"),
                pl.col("values/quantiles").struct.field("values/quantile/0.4").alias("values/quantile/40"),
                pl.col("values/quantiles").struct.field("values/quantile/0.5").alias("values/quantile/50"),
                pl.col("values/quantiles").struct.field("values/quantile/0.6").alias("values/quantile/60"),
                pl.col("values/quantiles").struct.field("values/quantile/0.7").alias("values/quantile/70"),
                pl.col("values/quantiles").struct.field("values/quantile/0.8").alias("values/quantile/80"),
                pl.col("values/quantiles").struct.field("values/quantile/0.9").alias("values/quantile/90"),
                pl.col("values/quantiles").struct.field("values/quantile/1").alias("values/quantile/100"),
                (pl.col("values/n_occurrences") > 0).alias("code/has_value"),
                (pl.col("values/sum") / pl.col("values/n_occurrences")).alias("values/mean"),
                pl.lit(self.config.default_value_sampling_strategy).alias("values/strategy"),
                pl.lit([]).cast(pl.List(pl.List(pl.Float64))).alias("values/range_options"),
            )
            .with_columns(
                (
                    (pl.col("values/sum_sqd") / pl.col("values/n_occurrences"))
                    - (pl.col("values/mean")) ** 2
                ).alias("values/variance"),
            )
            .with_columns(
                pl.col("values/variance").sqrt().alias("values/std"),
            )
        )

    def _set_data_at_code(self, code, col, value):
        code = code.lower()
        self.metadata = self.metadata.with_columns(
            pl.when(pl.col("code").str.to_lowercase() == code)
            .then(pl.lit(value))
            .otherwise(pl.col(col))
            .alias(col)
        )

    def _get_data_at_code(self, code, col):
        code = code.lower()
        return (
            self.metadata.filter(pl.col("code").str.to_lowercase() == code)
            .select(col)
            .item()
        )

    def _validate_codes(self, codes):
        valid_codes = {x.lower() for x in self.metadata["code"].to_list()}
        for x in codes:
            x = x.lower()
            if x not in valid_codes:
                raise ValueError(
                    f"Code '{x}' is not found in metadata"
                )
        return

    def _validate_range_bound(self, x):
        if isinstance(x, str):
            if not x.startswith("Q"):
                raise ValueError(
                    f"String value '{x}' start with Q followed by the quantile."
                )
            if float(x.replace("Q", "")) not in self.config.quantiles:
                raise ValueError(
                    f"Quantile '{x}' not supported, options are {self.config.quantiles}."
                )
        elif not isinstance(x, (int, float)):
            raise ValueError(f"Value '{x}' must be an int, float, or str.")

    def set_codes(self, codes: list[str] = None):
        if codes is None or not codes:
            self.code_options = self.metadata
        else:
            self._validate_codes(codes)
            codes = [x.lower() for x in codes]
            self.code_options = self.metadata.filter(
                pl.col("code").str.to_lowercase().is_in(codes)
            )
        self.code_options = self.code_options.with_columns(
            (pl.col("code/n_occurrences") / pl.col("code/n_occurrences").sum()).alias(
                "code/frequency"
            )
        )

    def set_values(self, strategy: str, data: list | dict):
        assert strategy in self.value_strategies
        if strategy == "manual":
            assert isinstance(data, dict) or isinstance(
                data, omegaconf.dictconfig.DictConfig
            )
            codes = data.keys()
        else:
            assert isinstance(data, list) or isinstance(
                data, omegaconf.listconfig.ListConfig
            )
            codes = data

        self._validate_codes(codes)

        for code in codes:
            self._set_data_at_code(code=code, col="values/strategy", value=strategy)
            if strategy == "manual":
                ranges = []
                for lower, upper in data[code]:
                    self._validate_range_bound(lower)
                    self._validate_range_bound(upper)
                    if isinstance(lower, str):
                        lower_quantile = int(lower.replace("Q", ""))
                        lower = self._get_data_at_code(
                            code=code, col=f"values/quantile/{lower_quantile}"
                        )
                    if isinstance(upper, str):
                        upper_quantile = int(upper.replace("Q", ""))
                        upper = self._get_data_at_code(
                            code=code, col=f"values/quantile/{upper_quantile}"
                        )
                    assert lower <= upper
                    # normalize the range based on mean/std from metadata
                    ranges.append([float(lower), float(upper)])
                self._set_data_at_code(
                    code=code, col="values/range_options", value=ranges
                )

        # refresh code options with updated value info
        self.set_codes(codes=self.code_options["code"].to_list())

    def sample_code(self):
        if self.code_options.height == 1:
            options = self.code_options
        else:
            match self.config.code_sampling_strategy:
                case "uniform":
                    options = self.code_options
                case "frequency":
                    num_buckets = int(
                        self.code_options["code/frequency"]
                        .log(base=10)
                        .floor()
                        .to_numpy()
                        .min()
                    )
                    bucket = np.random.choice([*range(num_buckets, 0)])
                    lower, upper = np.logspace(bucket, bucket + 1, 2)
                    options = self.code_options.filter(
                        pl.col("code/frequency").is_between(
                            lower_bound=lower, upper_bound=upper
                        )
                    )
        code = options.sample().to_dicts()[0]
        return code

    def sample_value_range(self, code):
        match code["values/strategy"]:
            case "manual":
                lower, upper = random.choice(code["values/range_options"])
            case "random_quantile":
                lower_quantile, upper_quantile = sorted(
                    random.sample(self.config.quantiles, 2)
                )
                lower = code[f"values/quantile/{int(lower_quantile)}"]
                upper = code[f"values/quantile/{int(upper_quantile)}"]
            case "random_normal":
                # random interval from the support of the normal distribution sampled according to its density
                def _normal_support(mu, sigma):
                    return scipy.stats.norm.ppf(np.random.rand(), loc=mu, scale=sigma)

                mu, sigma = code["values/mean"], code["values/std"]
                lower, upper = sorted(
                    [_normal_support(mu, sigma), _normal_support(mu, sigma)]
                )
        return lower, upper

    def sample_event(self):
        code = self.sample_code()
        if code["code/has_value"] and code["values/strategy"] != "ignore":
            use_value = True
            lower, upper = self.sample_value_range(code)
        else:
            use_value = False
            lower, upper = 0, 0  # mask later if needed
        event = {
            "name": code["code"],
            "code": code["code/vocab_index"],
            "has_value": code["code/has_value"],
            "use_value": use_value,
            "range_lower": lower,
            "range_upper": upper,
        }
        return event

    def normalize(self, query):
        if self.config.normalize_query:
            query["duration"] = (query["duration"] - self.config.min_future) / (
                self.config.max_future - self.config.min_future
            )
            query["offset"] = (query["offset"] - self.config.min_future) / (
                self.config.max_future - self.config.min_future
            )
            # tbd: query['range_lower'], query['range_upper']
            # range is provided in un-normalized units by user
            # can use mean/std from metadata
            # change in the preprocessing
        return query

    def sample_future(self, max_record_future):
        if max_record_future < 0:
            raise ValueError(
                f"max_record_future must be non-negative, but got {max_record_future}"
            )

        duration = self.sample_duration(max_record_future)
        offset = self.sample_offset(max(max_record_future-duration, 0))

        if (duration + offset) > max_record_future: 
            is_censored = True
        else:
            is_censored = False

        future = {"offset": offset, "duration": duration}

        return future, is_censored

    def sample_duration(self, max_record_future):
        match self.config.duration_sampling_strategy:
            case "within_record":
                if max_record_future <= self.config.min_duration:
                    duration = max_record_future
                else:
                    duration = np.random.randint(
                        low=self.config.min_duration,
                        high=min(self.config.max_duration, max_record_future),
                    )
            case "random":
                duration = np.random.randint(
                    self.config.min_duration, self.config.max_duration
                )
            case "categorical": 
                duration = np.random.choice(self.config.categorical_duration)
            case "fixed":
                duration = self.config.fixed_duration
        if duration < 0:
            raise ValueError(f"duration must be non-negative, but got {duration}")
        return duration

    def sample_offset(self, max_record_future):
        match self.config.offset_sampling_strategy:
            case "within_record":
                if max_record_future <= self.config.min_offset:
                    offset = max_record_future
                else:
                    offset = np.random.randint(
                        low=self.config.min_offset,
                        high=min(self.config.max_offset, max_record_future),
                    )
            case "random":
                offset = np.random.randint(self.config.min_offset, self.config.max_offset)
            case "categorical": 
                offset = np.random.choice(self.config.categorical_offset)
            case "fixed":
                offset = self.config.fixed_offset
        if offset < 0:
            raise ValueError(f"offset must be non-negative, but got {offset}")
        return offset

    def tally_answer(self, future_dynamic, query):
        time_delta = future_dynamic.tensors["dim0/time_delta_days"] * 1440
        if np.isnan(time_delta[0]):
            time_delta[0] = 0
        times = np.cumsum(time_delta)

        start_time = query["offset"]
        end_time = query["offset"] + query["duration"]
        start_idx = np.min(np.argwhere((times) >= start_time))
        if end_time >= times[-1]:
            end_idx = None
        else:
            end_idx = np.min(np.argwhere((times) > end_time))
            # end_idx is the first index you can't use
            assert start_idx <= end_idx

        if start_idx == end_idx:
            # query is short and fits between two measurements, ie. no data is observed
            return 0
        else:
            future_dynamic = future_dynamic[start_idx:end_idx]

        count = 0 
        code_idx = future_dynamic.tensors["dim1/code"] == query["code"]
        if query["has_value"] and query["use_value"]:
            values = future_dynamic.tensors["dim1/numeric_value"][code_idx]
            count = sum([query["range_lower"] <= x <= query["range_upper"]  for x in values])
        else: 
            count = sum(code_idx)

        return count

    def get_subject_times(self, subject_id):
        """
        alternative option to compute times
        time_delta = dynamic["dim0/time_delta_days"] * 1440
        if np.isnan(time_delta[0]):
            time_delta[0] = 0
        times = np.cumsum(time_delta)
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()
        times = static_row['time'].item().to_numpy().astype("datetime64[m]")
        assert np.all(times[:-1] <= times[1:]), 'subject times not sorted'
        return times

    def get_future_duration(self, subject_id, context_end_idx, record_end_idx):
        assert context_end_idx <= record_end_idx, f"context_end_idx: {context_end_idx}, record_end_idx: {record_end_idx}"
        times = self.get_subject_times(subject_id)
        # should be the timestamp at which the context ends (and not the timestamp of the next event)
        context_end_time = times[context_end_idx - 1]
        # should be the last timestamp included in the record
        # not the first timestamp after the end of the record
        # record_end_time is last time you can use, and not the first time you can't use
        record_end_time = times[record_end_idx - 1]
        future_duration = (record_end_time - context_end_time) / np.timedelta64(1, "m")
        return future_duration

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx: int) -> dict[str, list[float]]:
        
        context = super()._seeded_getitem(idx)

        subj_dynamic, subject_id, record_start_idx, record_end_idx = super().load_subject_dynamic_data(idx)

        future_duration = self.get_future_duration(subject_id, context["end_idx"], record_end_idx)
        future, is_censored = self.sample_future(max_record_future=future_duration)

        event = self.sample_event()
        query = future | event

        if is_censored:
            # censored is the mask for count and occurs
            answer = {"censored": is_censored, "count": -1, "occurs": -1} 
        else:
            future_dynamic = subj_dynamic[context["end_idx"] : record_end_idx]
            count = self.tally_answer(future_dynamic, query)
            answer = {"censored": is_censored, "count": count, "occurs": count != 0}

        query = self.normalize(query)

        item = {"context": context, "query": query, "answer": answer}

        return item

    def _query_collate(self, batch: list[dict]) -> dict:
        return {
            "offset": torch.tensor([x["offset"] for x in batch], dtype=torch.float64),
            "duration": torch.tensor([x["duration"] for x in batch], dtype=torch.float64),
            "code": torch.tensor([x["code"] for x in batch], dtype=torch.int64),
            "has_value": torch.tensor([x["has_value"] for x in batch], dtype=torch.bool),
            "use_value": torch.tensor([x["use_value"] for x in batch], dtype=torch.bool),
            "range_lower": torch.tensor([x["range_lower"] for x in batch], dtype=torch.float64),
            "range_upper": torch.tensor([x["range_upper"] for x in batch], dtype=torch.float64),
        }

    def _answer_collate(self, batch: list[dict]) -> dict:
        return {
            "censored": torch.tensor([x["censored"] for x in batch], dtype=torch.bool).unsqueeze(1),
            "count": torch.tensor([x["count"] for x in batch], dtype=torch.float64).unsqueeze(1), # count except -1 for censored
            "occurs": torch.tensor([x["occurs"] for x in batch], dtype=torch.float64).unsqueeze(1), # bool except -1 for censored
        }

    def collate(self, batch: list[dict]) -> dict:
        return {
            "context": super().collate([x["context"] for x in batch]),
            "query": self._query_collate([x["query"] for x in batch]),
            "answer": self._answer_collate([x["answer"] for x in batch]),
        }
