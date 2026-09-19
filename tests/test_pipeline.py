import logging
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.pipeline.preprocess import (
    SetAPreprocessor,
    load_set_a_feature_spec,
    prepare_set_a_data,
)
from src.utils import logger as logger_module
from src.utils.logger import get_project_logger


def _hmda_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_taken": [1, 2, 3, 4],
            "applicant_race-1": [1, 2, 1, 3],
            "applicant_race-2": [2, None, 1, None],
            "applicant_race-3": [None, None, None, None],
            "applicant_race-4": [None, None, None, None],
            "applicant_race-5": [None, None, None, None],
            "applicant_sex": [1, 2, 1, 2],
            "income": [100.0, None, 80.0, 50.0],
            "loan_amount": [200.0, 150.0, 120.0, 90.0],
            "loan_purpose": [1, 1, 2, 2],
            "lien_status": [1, 1, 1, 2],
            "loan_type": [1, 2, 1, 2],
        }
    )


class TestSetAPreprocessing(unittest.TestCase):
    def test_prepare_set_a_data_filters_actions_and_prevents_target_leakage(self):
        prepared = prepare_set_a_data(_hmda_frame())

        self.assertEqual(prepared.X.index.tolist(), [0, 1, 2])
        self.assertEqual(prepared.y.tolist(), [1, 1, 0])
        self.assertEqual(prepared.y.name, "y_approved")
        self.assertNotIn("action_taken", prepared.X.columns)
        self.assertEqual(
            prepared.X.columns.tolist(),
            [
                "applicant_race-1",
                "applicant_race-2",
                "applicant_race-3",
                "applicant_race-4",
                "applicant_race-5",
                "applicant_sex",
                "income",
                "loan_amount",
                "loan_purpose",
                "lien_status",
                "loan_type",
            ],
        )

    def test_set_a_preprocessor_multihot_encodes_race_and_preserves_groups(self):
        prepared = prepare_set_a_data(_hmda_frame())
        processor = SetAPreprocessor().fit(prepared.X, prepared.y)

        transformed = processor.transform(prepared.X)
        names = transformed.columns.tolist()

        self.assertIn("race__applicant_race__1", names)
        self.assertIn("race__applicant_race__2", names)
        self.assertEqual(transformed.loc[0, "race__applicant_race__1"], 1)
        self.assertEqual(transformed.loc[0, "race__applicant_race__2"], 1)
        self.assertEqual(transformed.loc[1, "race__applicant_race__2"], 1)
        self.assertNotIn("action_taken", names)

        groups = processor.explanation_groups()
        self.assertEqual(
            set(groups),
            {
                "applicant_race",
                "applicant_sex",
                "applicant_income",
                "loan_amount",
                "loan_purpose",
                "lien_status",
                "loan_type",
            },
        )
        self.assertEqual(
            set(groups["applicant_race"]),
            {"race__applicant_race__1", "race__applicant_race__2"},
        )
        self.assertEqual(set().union(*map(set, groups.values())), set(names))

    def test_set_a_preprocessor_rejects_missing_locked_columns(self):
        prepared = prepare_set_a_data(_hmda_frame())
        with self.assertRaisesRegex(ValueError, "loan_type"):
            SetAPreprocessor().fit(prepared.X.drop(columns="loan_type"), prepared.y)

    def test_set_a_contract_rejects_an_invalid_raw_input_list(self):
        spec = load_set_a_feature_spec()
        invalid_spec = replace(spec, raw_columns=spec.raw_columns[:-1])

        with self.assertRaisesRegex(ValueError, "locked 11 raw model inputs"):
            prepare_set_a_data(_hmda_frame(), invalid_spec)

    def test_set_a_contract_rejects_incorrect_action_partition(self):
        spec = load_set_a_feature_spec()
        invalid_spec = replace(spec, eligible_values=(1, 2, 3, 4))

        with self.assertRaisesRegex(ValueError, "eligible action codes"):
            prepare_set_a_data(_hmda_frame(), invalid_spec)

    def test_preprocessing_writes_informative_runtime_logs(self):
        with self.assertLogs("src.pipeline.preprocess", level="INFO") as captured:
            prepared = prepare_set_a_data(_hmda_frame())
            SetAPreprocessor().fit_transform(prepared.X, prepared.y)

        logs = "\n".join(captured.output)
        self.assertIn("Prepared Set A data: eligible_rows=3", logs)
        self.assertIn("Fitted Set A preprocessor", logs)
        self.assertIn("Transformed Set A data", logs)

    def test_project_logger_adds_one_rotating_file_handler_and_writes_to_it(self):
        logger_name = "tests.set_a_file_logging"
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        with TemporaryDirectory() as temporary_directory:
            try:
                with patch.object(logger_module, "REPORTS_DIR", Path(temporary_directory)):
                    configured = get_project_logger(logger_name)
                    configured = get_project_logger(logger_name, "set_a.log")
                    configured = get_project_logger(logger_name, "set_a.log")
                    configured.info("Set A file-log smoke test")

                log_path = Path(temporary_directory) / "logs" / "set_a.log"
                self.assertTrue(log_path.exists())
                self.assertIn("Set A file-log smoke test", log_path.read_text(encoding="utf-8"))
                file_handlers = [
                    handler
                    for handler in configured.handlers
                    if isinstance(handler, logging.FileHandler)
                ]
                self.assertEqual(len(file_handlers), 1)
            finally:
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()
