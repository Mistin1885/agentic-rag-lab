"""Mock data-processing tools exposed to the Gemini agent."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from agentic_rag_lab.examples_data.mock_orders import RAW_DATASETS


Money = Decimal


def _money(value: Money) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _parse_decimal(value: Any, default: Decimal | None = None) -> Decimal:
    if value in ("", None):
        if default is not None:
            return default
        raise ValueError("empty decimal value")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc


def _parse_int(value: Any) -> int:
    if value in ("", None):
        raise ValueError("empty integer value")
    return int(str(value))


def _dataset(dataset_id: str) -> list[dict[str, Any]]:
    if dataset_id not in RAW_DATASETS:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    return deepcopy(RAW_DATASETS[dataset_id])


def inspect_schema(dataset_id: str) -> dict[str, Any]:
    """Inspect columns, row counts, missing values, and rough data types."""
    rows = _dataset(dataset_id)
    columns = sorted({key for row in rows for key in row})
    missing = {column: 0 for column in columns}
    type_examples: dict[str, str] = {}

    for row in rows:
        for column in columns:
            value = row.get(column)
            if value in ("", None):
                missing[column] += 1
            elif column not in type_examples:
                type_examples[column] = type(value).__name__

    return {
        "dataset_id": dataset_id,
        "row_count": len(rows),
        "columns": columns,
        "missing_values": missing,
        "type_examples": type_examples,
        "validation": {
            "ok": len(rows) > 0 and bool(columns),
            "checks": ["dataset_exists", "has_rows", "has_columns"],
        },
    }


def clean_orders(dataset_id: str) -> dict[str, Any]:
    """Normalize raw order rows and produce data-quality warnings."""
    raw_rows = _dataset(dataset_id)
    clean_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    required = ["order_id", "date", "customer", "sku", "quantity", "unit_price", "status"]
    seen_ids: set[str] = set()

    for index, row in enumerate(raw_rows):
        row_warnings: list[str] = []
        missing_required = [field for field in required if row.get(field) in ("", None)]
        if missing_required:
            row_warnings.append(f"missing required fields: {', '.join(missing_required)}")

        order_id = str(row.get("order_id", "")).strip()
        if order_id in seen_ids:
            row_warnings.append("duplicate order_id")
        seen_ids.add(order_id)

        if row.get("discount_pct") in ("", None):
            row_warnings.append("missing discount_pct; defaulted to 0")
        if row.get("region") in ("", None):
            row_warnings.append("missing region; defaulted to unknown")

        try:
            quantity = _parse_int(row.get("quantity"))
            unit_price = _parse_decimal(row.get("unit_price"))
            discount_pct = _parse_decimal(row.get("discount_pct"), default=Decimal("0"))
            date.fromisoformat(str(row.get("date")))
        except (TypeError, ValueError) as exc:
            warnings.append(
                {
                    "row_index": index,
                    "order_id": order_id or None,
                    "issue": str(exc),
                    "action": "dropped",
                }
            )
            continue

        if quantity <= 0:
            row_warnings.append("non-positive quantity")
        if discount_pct < 0 or discount_pct > Decimal("0.70"):
            row_warnings.append("discount outside expected range")

        status = str(row.get("status", "")).strip().lower()
        gross = Decimal(quantity) * unit_price
        net = gross * (Decimal("1") - discount_pct)
        if status == "refunded":
            net = -abs(net)

        cleaned = {
            "order_id": order_id,
            "date": str(row["date"]),
            "customer": str(row.get("customer", "")).strip(),
            "segment": str(row.get("segment", "unknown")).strip() or "unknown",
            "region": str(row.get("region", "unknown")).strip() or "unknown",
            "channel": str(row.get("channel", "unknown")).strip() or "unknown",
            "sku": str(row.get("sku", "")).strip(),
            "quantity": quantity,
            "unit_price": _money(unit_price),
            "discount_pct": float(discount_pct),
            "status": status,
            "net_revenue": _money(net),
        }
        clean_rows.append(cleaned)

        for issue in row_warnings:
            warnings.append(
                {
                    "row_index": index,
                    "order_id": order_id,
                    "issue": issue,
                    "action": "kept_for_analysis",
                }
            )

    return {
        "dataset_id": dataset_id,
        "clean_rows": clean_rows,
        "input_count": len(raw_rows),
        "clean_count": len(clean_rows),
        "dropped_count": len(raw_rows) - len(clean_rows),
        "warnings": warnings,
        "validation": {
            "ok": len(clean_rows) > 0,
            "checks": ["types_normalized", "net_revenue_computed", "warnings_collected"],
        },
    }


def compute_kpis(
    dataset_id: str,
    include_by_region: bool = True,
    include_by_channel: bool = True,
) -> dict[str, Any]:
    """Compute business KPIs from normalized orders."""
    cleaned = clean_orders(dataset_id)
    rows = cleaned["clean_rows"]
    paid_rows = [row for row in rows if row["status"] == "paid"]
    refund_rows = [row for row in rows if row["status"] == "refunded"]

    total_revenue = sum(Decimal(str(row["net_revenue"])) for row in rows)
    paid_revenue = sum(Decimal(str(row["net_revenue"])) for row in paid_rows)
    refund_value = sum(abs(Decimal(str(row["net_revenue"]))) for row in refund_rows)

    by_region: dict[str, Money] = defaultdict(lambda: Decimal("0"))
    by_channel: dict[str, Money] = defaultdict(lambda: Decimal("0"))
    by_customer: dict[str, Money] = defaultdict(lambda: Decimal("0"))
    sku_counter: Counter[str] = Counter()

    for row in rows:
        revenue = Decimal(str(row["net_revenue"]))
        by_region[row["region"]] += revenue
        by_channel[row["channel"]] += revenue
        by_customer[row["customer"]] += revenue
        sku_counter[row["sku"]] += row["quantity"]

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "order_count": len(rows),
        "paid_order_count": len(paid_rows),
        "refund_count": len(refund_rows),
        "total_revenue": _money(total_revenue),
        "paid_revenue": _money(paid_revenue),
        "refund_value": _money(refund_value),
        "average_order_value": _money(total_revenue / Decimal(len(rows))) if rows else 0.0,
        "top_sku_by_quantity": sku_counter.most_common(1)[0][0] if sku_counter else None,
        "top_customers_by_revenue": [
            {"customer": customer, "revenue": _money(revenue)}
            for customer, revenue in sorted(by_customer.items(), key=lambda item: item[1], reverse=True)
        ],
        "data_quality_warning_count": len(cleaned["warnings"]),
        "validation": {
            "ok": True,
            "checks": ["clean_orders_completed", "revenue_reconciled", "refunds_included"],
        },
    }

    if include_by_region:
        result["revenue_by_region"] = {
            region: _money(revenue)
            for region, revenue in sorted(by_region.items(), key=lambda item: item[0])
        }
    if include_by_channel:
        result["revenue_by_channel"] = {
            channel: _money(revenue)
            for channel, revenue in sorted(by_channel.items(), key=lambda item: item[0])
        }

    return result


def detect_risk_signals(dataset_id: str, sensitivity: str = "medium") -> dict[str, Any]:
    """Detect unusual discounts, refunds, missing fields, and negative quantities."""
    cleaned = clean_orders(dataset_id)
    rows = cleaned["clean_rows"]
    sensitivity_thresholds = {
        "low": Decimal("0.60"),
        "medium": Decimal("0.30"),
        "high": Decimal("0.15"),
    }
    threshold = sensitivity_thresholds.get(sensitivity, Decimal("0.30"))
    signals: list[dict[str, Any]] = []
    customer_signal_counts: Counter[str] = Counter()

    for row in rows:
        reasons: list[str] = []
        discount = Decimal(str(row["discount_pct"]))
        if discount >= threshold:
            reasons.append(f"discount >= {threshold}")
        if row["status"] == "refunded":
            reasons.append("refunded order")
        if row["quantity"] <= 0:
            reasons.append("non-positive quantity")
        if row["region"] == "unknown":
            reasons.append("missing region")

        if reasons:
            customer_signal_counts[row["customer"]] += len(reasons)
            signals.append(
                {
                    "order_id": row["order_id"],
                    "customer": row["customer"],
                    "net_revenue": row["net_revenue"],
                    "reasons": reasons,
                }
            )

    for warning in cleaned["warnings"]:
        if warning["action"] == "kept_for_analysis":
            customer = next(
                (row["customer"] for row in rows if row["order_id"] == warning["order_id"]),
                "unknown",
            )
            customer_signal_counts[customer] += 1

    return {
        "dataset_id": dataset_id,
        "sensitivity": sensitivity,
        "risk_signal_count": len(signals),
        "signals": signals,
        "customers_to_follow_up": [
            {"customer": customer, "signal_count": count}
            for customer, count in customer_signal_counts.most_common()
        ],
        "validation": {
            "ok": True,
            "checks": ["discount_threshold_applied", "refunds_checked", "data_quality_warnings_checked"],
        },
    }


def validate_answer(
    dataset_id: str,
    claimed_metrics: dict[str, Any] | None = None,
    required_sections: list[str] | None = None,
) -> dict[str, Any]:
    """Validate claimed metrics against deterministic tool calculations."""
    claimed_metrics = claimed_metrics or {}
    required_sections = required_sections or []
    kpis = compute_kpis(dataset_id)
    risks = detect_risk_signals(dataset_id)
    expected = {
        "order_count": kpis["order_count"],
        "paid_order_count": kpis["paid_order_count"],
        "refund_count": kpis["refund_count"],
        "total_revenue": kpis["total_revenue"],
        "paid_revenue": kpis["paid_revenue"],
        "refund_value": kpis["refund_value"],
        "risk_signal_count": risks["risk_signal_count"],
    }
    mismatches = []

    for key, expected_value in expected.items():
        if key not in claimed_metrics:
            continue
        claimed_value = claimed_metrics[key]
        if isinstance(expected_value, float):
            matches = abs(float(claimed_value) - expected_value) < 0.01
        else:
            matches = claimed_value == expected_value
        if not matches:
            mismatches.append(
                {
                    "metric": key,
                    "claimed": claimed_value,
                    "expected": expected_value,
                }
            )

    missing_sections = [section for section in required_sections if not section.strip()]

    return {
        "dataset_id": dataset_id,
        "expected_metrics": expected,
        "claimed_metrics_checked": sorted(claimed_metrics.keys()),
        "mismatches": mismatches,
        "missing_sections": missing_sections,
        "validation": {
            "ok": not mismatches and not missing_sections,
            "checks": ["metrics_recomputed", "claimed_metrics_compared", "required_sections_checked"],
        },
    }


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "inspect_schema",
        "description": "Inspect the raw dataset columns, row count, missing values, and type examples. Use this before cleaning if the user asks about data quality or schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "enum": ["demo_orders_q2"],
                    "description": "The dataset to inspect.",
                }
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "clean_orders",
        "description": "Normalize raw order data, convert numeric fields, compute net revenue, and return data-quality warnings.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "enum": ["demo_orders_q2"],
                    "description": "The dataset to clean.",
                }
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "compute_kpis",
        "description": "Compute revenue, refunds, average order value, top SKU, and customer/region/channel summaries.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "enum": ["demo_orders_q2"],
                    "description": "The dataset to analyze.",
                },
                "include_by_region": {
                    "type": "boolean",
                    "description": "Whether to include revenue grouped by region.",
                },
                "include_by_channel": {
                    "type": "boolean",
                    "description": "Whether to include revenue grouped by channel.",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "detect_risk_signals",
        "description": "Find suspicious orders and customers needing follow-up, using discounts, refunds, missing fields, and negative quantities.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "enum": ["demo_orders_q2"],
                    "description": "The dataset to scan.",
                },
                "sensitivity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Risk sensitivity. high catches more issues; low catches only severe issues.",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "validate_answer",
        "description": "Validate key metrics that will appear in the final answer against deterministic tool calculations. Use this before the final response.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "enum": ["demo_orders_q2"],
                    "description": "The dataset used for validation.",
                },
                "claimed_metrics": {
                    "type": "object",
                    "description": "Metrics the final answer plans to cite, such as total_revenue, order_count, refund_count, risk_signal_count.",
                },
                "required_sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Section names expected in the final answer.",
                },
            },
            "required": ["dataset_id"],
        },
    },
]

TOOL_REGISTRY = {
    "inspect_schema": inspect_schema,
    "clean_orders": clean_orders,
    "compute_kpis": compute_kpis,
    "detect_risk_signals": detect_risk_signals,
    "validate_answer": validate_answer,
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool with basic error capture."""
    if name not in TOOL_REGISTRY:
        return {
            "error": f"unknown tool: {name}",
            "validation": {"ok": False, "checks": ["tool_exists"]},
        }
    try:
        return TOOL_REGISTRY[name](**args)
    except Exception as exc:  # pragma: no cover - defensive boundary for demo logs.
        return {
            "error": str(exc),
            "validation": {"ok": False, "checks": ["tool_execution"]},
        }
