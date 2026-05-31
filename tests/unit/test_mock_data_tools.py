from __future__ import annotations

import unittest

from agentic_rag_lab.tools.mock_data_tools import (
    clean_orders,
    compute_kpis,
    detect_risk_signals,
    inspect_schema,
    validate_answer,
)


class ToolTests(unittest.TestCase):
    def test_inspect_schema(self) -> None:
        result = inspect_schema("demo_orders_q2")

        self.assertTrue(result["validation"]["ok"])
        self.assertEqual(result["row_count"], 10)
        self.assertIn("order_id", result["columns"])
        self.assertEqual(result["missing_values"]["region"], 1)

    def test_clean_orders_keeps_quality_warnings(self) -> None:
        result = clean_orders("demo_orders_q2")

        self.assertTrue(result["validation"]["ok"])
        self.assertEqual(result["input_count"], 10)
        self.assertEqual(result["clean_count"], 10)
        self.assertGreaterEqual(len(result["warnings"]), 4)

    def test_compute_kpis_reconciles_expected_metrics(self) -> None:
        result = compute_kpis("demo_orders_q2")

        self.assertEqual(result["order_count"], 10)
        self.assertEqual(result["refund_count"], 1)
        self.assertEqual(result["total_revenue"], 66045.0)
        self.assertEqual(result["paid_revenue"], 66845.0)
        self.assertEqual(result["refund_value"], 800.0)
        self.assertEqual(result["top_sku_by_quantity"], "pro-license")

    def test_detect_risk_signals(self) -> None:
        result = detect_risk_signals("demo_orders_q2", sensitivity="medium")

        self.assertTrue(result["validation"]["ok"])
        self.assertEqual(result["risk_signal_count"], 5)
        customers = {row["customer"] for row in result["customers_to_follow_up"]}
        self.assertIn("Blue Pine", customers)
        self.assertIn("Contoso Health", customers)

    def test_validate_answer_catches_mismatch(self) -> None:
        result = validate_answer(
            "demo_orders_q2",
            claimed_metrics={"total_revenue": 1, "order_count": 10},
            required_sections=["summary"],
        )

        self.assertFalse(result["validation"]["ok"])
        self.assertEqual(result["mismatches"][0]["metric"], "total_revenue")

    def test_validate_answer_accepts_expected_metrics(self) -> None:
        result = validate_answer(
            "demo_orders_q2",
            claimed_metrics={
                "total_revenue": 66045.0,
                "order_count": 10,
                "refund_count": 1,
                "risk_signal_count": 5,
            },
            required_sections=["summary", "validation"],
        )

        self.assertTrue(result["validation"]["ok"])


if __name__ == "__main__":
    unittest.main()
