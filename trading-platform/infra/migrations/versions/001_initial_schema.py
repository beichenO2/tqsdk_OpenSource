"""Initial schema — all ORM models.

Revision ID: 001_initial
Revises:
Create Date: 2026-04-10
"""

from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    asset_class_enum = sa.Enum("futures", "crypto", "stock", "option", name="asset_class_enum")
    order_side_enum = sa.Enum("buy", "sell", name="order_side_enum")
    order_type_enum = sa.Enum("market", "limit", "stop", "stop_limit", "fok", "fak", name="order_type_enum")
    offset_flag_enum = sa.Enum("open", "close", "close_today", "close_yesterday", name="offset_flag_enum")
    order_status_enum = sa.Enum(
        "pending", "submitted", "partial_filled", "filled",
        "cancelled", "rejected", "expired",
        name="order_status_enum",
    )
    position_side_enum = sa.Enum("long", "short", name="position_side_enum")
    strategy_type_enum = sa.Enum("rule_based", "ml_supervised", "rl", "hybrid", name="strategy_type_enum")
    strategy_status_enum = sa.Enum(
        "draft", "backtesting", "paper_trading", "live", "paused", "retired",
        name="strategy_status_enum",
    )
    risk_rule_type_enum = sa.Enum(
        "max_position", "max_order_size", "max_drawdown",
        "daily_loss_limit", "order_rate_limit", "concentration_limit", "custom",
        name="risk_rule_type_enum",
    )
    risk_severity_enum = sa.Enum("info", "warning", "critical", "block", name="risk_severity_enum")
    backtest_status_enum = sa.Enum("queued", "running", "completed", "failed", "cancelled", name="backtest_status_enum")
    evidence_type_enum = sa.Enum(
        "signal", "risk_check", "order_decision", "position_change",
        "model_prediction", "feature_value", "manual_override",
        name="evidence_type_enum",
    )
    data_source_type_enum = sa.Enum(
        "tqsdk", "exchange_ws", "exchange_rest", "csv_file",
        "parquet_file", "duckdb", "external_api",
        name="data_source_type_enum",
    )
    data_frequency_enum = sa.Enum(
        "tick", "1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w",
        name="data_frequency_enum",
    )
    model_framework_enum = sa.Enum(
        "pytorch", "xgboost", "lightgbm", "stable_baselines3", "sklearn", "custom",
        name="model_framework_enum",
    )
    model_status_enum = sa.Enum(
        "training", "trained", "validating", "deployed", "archived", "failed",
        name="model_status_enum",
    )

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(64), unique=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.Text, nullable=False),
        sa.Column("display_name", sa.String(128), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_superuser", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # --- exchanges ---
    op.create_table(
        "exchanges",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("code", sa.String(16), unique=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("asset_class", asset_class_enum, nullable=False),
        sa.Column("timezone", sa.String(64), server_default="Asia/Shanghai"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- instruments ---
    op.create_table(
        "instruments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("exchange_code", sa.String(16), nullable=False),
        sa.Column("asset_class", asset_class_enum, nullable=False),
        sa.Column("name", sa.String(128), nullable=True),
        sa.Column("currency", sa.String(8), server_default="CNY"),
        sa.Column("tick_size", sa.Numeric(18, 8), nullable=False),
        sa.Column("lot_size", sa.Numeric(18, 8), server_default="1"),
        sa.Column("multiplier", sa.Numeric(18, 4), server_default="1"),
        sa.Column("margin_rate", sa.Numeric(8, 4), nullable=True),
        sa.Column("expire_date", sa.Date, nullable=True),
        sa.Column("underlying", sa.String(32), nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_instruments_symbol_exchange", "instruments", ["symbol", "exchange_code"], unique=True)
    op.create_index("ix_instruments_asset_class", "instruments", ["asset_class"])

    # --- api_keys ---
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("key_hash", sa.String(128), unique=True, nullable=False),
        sa.Column("scopes", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])

    # --- strategies ---
    op.create_table(
        "strategies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("slug", sa.String(128), unique=True, nullable=False),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_type", strategy_type_enum, nullable=False),
        sa.Column("status", strategy_status_enum, nullable=False, server_default="draft"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("target_instruments", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_strategies_owner_id", "strategies", ["owner_id"])
    op.create_index("ix_strategies_status", "strategies", ["status"])

    # --- strategy_versions ---
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("code_snapshot", sa.Text, nullable=True),
        sa.Column("config_json", sa.Text, nullable=True),
        sa.Column("changelog", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_strategy_versions_strategy_version", "strategy_versions", ["strategy_id", "version"], unique=True)

    # --- strategy_params ---
    op.create_table(
        "strategy_params",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("param_type", sa.String(16), server_default="string"),
        sa.Column("description", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_strategy_params_strategy_key", "strategy_params", ["strategy_id", "key"], unique=True)

    # --- orders ---
    op.create_table(
        "orders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("instrument_symbol", sa.String(32), nullable=False),
        sa.Column("exchange_code", sa.String(16), nullable=False),
        sa.Column("side", order_side_enum, nullable=False),
        sa.Column("order_type", order_type_enum, nullable=False),
        sa.Column("offset", offset_flag_enum, server_default="open"),
        sa.Column("price", sa.Numeric(18, 8), nullable=True),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("filled_quantity", sa.Numeric(18, 8), server_default="0"),
        sa.Column("avg_fill_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("status", order_status_enum, nullable=False, server_default="pending"),
        sa.Column("broker_order_id", sa.String(128), nullable=True),
        sa.Column("stop_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("time_in_force", sa.String(8), server_default="GTC"),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reject_reason", sa.Text, nullable=True),
        sa.Column("source", sa.String(16), server_default="manual"),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_orders_user_id", "orders", ["user_id"])
    op.create_index("ix_orders_strategy_id", "orders", ["strategy_id"])
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_symbol_exchange", "orders", ["instrument_symbol", "exchange_code"])
    op.create_index("ix_orders_created_at", "orders", ["created_at"])

    # --- fills ---
    op.create_table(
        "fills",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("price", sa.Numeric(18, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("commission", sa.Numeric(18, 8), server_default="0"),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("broker_fill_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_fills_order_id", "fills", ["order_id"])
    op.create_index("ix_fills_filled_at", "fills", ["filled_at"])

    # --- positions ---
    op.create_table(
        "positions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("instrument_symbol", sa.String(32), nullable=False),
        sa.Column("exchange_code", sa.String(16), nullable=False),
        sa.Column("side", position_side_enum, nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), server_default="0"),
        sa.Column("avg_entry_price", sa.Numeric(18, 8), server_default="0"),
        sa.Column("unrealized_pnl", sa.Numeric(18, 8), server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(18, 8), server_default="0"),
        sa.Column("margin_used", sa.Numeric(18, 8), server_default="0"),
        sa.Column("last_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_positions_user_instrument_side", "positions",
        ["user_id", "instrument_symbol", "exchange_code", "side"], unique=True,
    )
    op.create_index("ix_positions_strategy_id", "positions", ["strategy_id"])

    # --- position_snapshots ---
    op.create_table(
        "position_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("position_id", sa.String(36), sa.ForeignKey("positions.id"), nullable=False),
        sa.Column("snapshot_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("avg_entry_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("mark_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(18, 8), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(18, 8), nullable=False),
    )
    op.create_index("ix_position_snapshots_date", "position_snapshots", ["snapshot_date"])
    op.create_index("ix_position_snapshots_position_date", "position_snapshots", ["position_id", "snapshot_date"], unique=True)

    # --- risk_rules ---
    op.create_table(
        "risk_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("rule_type", risk_rule_type_enum, nullable=False),
        sa.Column("severity", risk_severity_enum, server_default="warning"),
        sa.Column("instrument_filter", sa.String(64), nullable=True),
        sa.Column("threshold_value", sa.Numeric(18, 8), nullable=False),
        sa.Column("cooldown_seconds", sa.Integer, server_default="0"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("config_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_risk_rules_user_id", "risk_rules", ["user_id"])
    op.create_index("ix_risk_rules_strategy_id", "risk_rules", ["strategy_id"])

    # --- risk_alerts ---
    op.create_table(
        "risk_alerts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("rule_id", sa.String(36), sa.ForeignKey("risk_rules.id"), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("severity", risk_severity_enum, nullable=False),
        sa.Column("triggered_value", sa.Numeric(18, 8), nullable=False),
        sa.Column("threshold_value", sa.Numeric(18, 8), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("acknowledged", sa.Boolean, server_default="false"),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_risk_alerts_user_id", "risk_alerts", ["user_id"])
    op.create_index("ix_risk_alerts_created_at", "risk_alerts", ["created_at"])
    op.create_index("ix_risk_alerts_severity", "risk_alerts", ["severity"])

    # --- backtest_runs ---
    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("strategy_version_id", sa.String(36), sa.ForeignKey("strategy_versions.id"), nullable=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("status", backtest_status_enum, nullable=False, server_default="queued"),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("initial_capital", sa.Numeric(18, 4), nullable=False),
        sa.Column("instruments_json", sa.Text, nullable=True),
        sa.Column("params_json", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("result_path", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_backtest_runs_user_id", "backtest_runs", ["user_id"])
    op.create_index("ix_backtest_runs_strategy_id", "backtest_runs", ["strategy_id"])
    op.create_index("ix_backtest_runs_status", "backtest_runs", ["status"])

    # --- backtest_trades ---
    op.create_table(
        "backtest_trades",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("instrument_symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("price", sa.Numeric(18, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("commission", sa.Numeric(18, 8), server_default="0"),
        sa.Column("pnl", sa.Numeric(18, 8), nullable=True),
        sa.Column("traded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signal_name", sa.String(64), nullable=True),
    )
    op.create_index("ix_backtest_trades_run_id", "backtest_trades", ["run_id"])

    # --- backtest_metrics ---
    op.create_table(
        "backtest_metrics",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("metric_value", sa.Numeric(18, 8), nullable=False),
    )
    op.create_index("ix_backtest_metrics_run_name", "backtest_metrics", ["run_id", "metric_name"], unique=True)

    # --- evidence_records ---
    op.create_table(
        "evidence_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("evidence_type", evidence_type_enum, nullable=False),
        sa.Column("instrument_symbol", sa.String(32), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_json", sa.Text, nullable=False),
        sa.Column("explanation", sa.Text, nullable=True),
        sa.Column("parent_id", sa.String(36), sa.ForeignKey("evidence_records.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_evidence_records_user_id", "evidence_records", ["user_id"])
    op.create_index("ix_evidence_records_order_id", "evidence_records", ["order_id"])
    op.create_index("ix_evidence_records_strategy_id", "evidence_records", ["strategy_id"])
    op.create_index("ix_evidence_records_type", "evidence_records", ["evidence_type"])
    op.create_index("ix_evidence_records_timestamp", "evidence_records", ["timestamp"])

    # --- decision_logs ---
    op.create_table(
        "decision_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("instrument_symbol", sa.String(32), nullable=True),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("evidence_ids_json", sa.Text, nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_decision_logs_user_id", "decision_logs", ["user_id"])
    op.create_index("ix_decision_logs_strategy_id", "decision_logs", ["strategy_id"])
    op.create_index("ix_decision_logs_decided_at", "decision_logs", ["decided_at"])

    # --- data_sources ---
    op.create_table(
        "data_sources",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), unique=True, nullable=False),
        sa.Column("source_type", data_source_type_enum, nullable=False),
        sa.Column("config_json", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("priority", sa.Integer, server_default="0"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- data_subscriptions ---
    op.create_table(
        "data_subscriptions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("data_source_id", sa.String(36), sa.ForeignKey("data_sources.id"), nullable=False),
        sa.Column("instrument_symbol", sa.String(32), nullable=False),
        sa.Column("exchange_code", sa.String(16), nullable=False),
        sa.Column("frequency", data_frequency_enum, nullable=False),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_data_subs_user_id", "data_subscriptions", ["user_id"])
    op.create_index(
        "ix_data_subs_unique", "data_subscriptions",
        ["user_id", "data_source_id", "instrument_symbol", "frequency"], unique=True,
    )

    # --- ml_models ---
    op.create_table(
        "ml_models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("slug", sa.String(128), unique=True, nullable=False),
        sa.Column("framework", model_framework_enum, nullable=False),
        sa.Column("strategy_id", sa.String(36), sa.ForeignKey("strategies.id"), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("target_variable", sa.String(128), nullable=True),
        sa.Column("feature_set_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_ml_models_strategy_id", "ml_models", ["strategy_id"])

    # --- ml_model_versions ---
    op.create_table(
        "ml_model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.String(36), sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("status", model_status_enum, nullable=False, server_default="training"),
        sa.Column("artifact_path", sa.String(512), nullable=True),
        sa.Column("metrics_json", sa.Text, nullable=True),
        sa.Column("hyperparams_json", sa.Text, nullable=True),
        sa.Column("training_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_ml_model_versions_model_version", "ml_model_versions", ["model_id", "version"], unique=True)
    op.create_index("ix_ml_model_versions_status", "ml_model_versions", ["status"])

    # --- ml_experiments ---
    op.create_table(
        "ml_experiments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("ml_model_versions.id"), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("config_json", sa.Text, nullable=True),
        sa.Column("results_json", sa.Text, nullable=True),
        sa.Column("dataset_ref", sa.String(512), nullable=True),
        sa.Column("train_score", sa.Numeric(10, 6), nullable=True),
        sa.Column("val_score", sa.Numeric(10, 6), nullable=True),
        sa.Column("test_score", sa.Numeric(10, 6), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_ml_experiments_model_version_id", "ml_experiments", ["model_version_id"])
    op.create_index("ix_ml_experiments_user_id", "ml_experiments", ["user_id"])


def downgrade() -> None:
    op.drop_table("ml_experiments")
    op.drop_table("ml_model_versions")
    op.drop_table("ml_models")
    op.drop_table("data_subscriptions")
    op.drop_table("data_sources")
    op.drop_table("decision_logs")
    op.drop_table("evidence_records")
    op.drop_table("backtest_metrics")
    op.drop_table("backtest_trades")
    op.drop_table("backtest_runs")
    op.drop_table("risk_alerts")
    op.drop_table("risk_rules")
    op.drop_table("position_snapshots")
    op.drop_table("positions")
    op.drop_table("fills")
    op.drop_table("orders")
    op.drop_table("strategy_params")
    op.drop_table("strategy_versions")
    op.drop_table("strategies")
    op.drop_table("api_keys")
    op.drop_table("instruments")
    op.drop_table("exchanges")
    op.drop_table("users")

    for enum_name in [
        "model_status_enum", "model_framework_enum",
        "data_frequency_enum", "data_source_type_enum",
        "evidence_type_enum", "backtest_status_enum",
        "risk_severity_enum", "risk_rule_type_enum",
        "strategy_status_enum", "strategy_type_enum",
        "position_side_enum", "order_status_enum",
        "offset_flag_enum", "order_type_enum",
        "order_side_enum", "asset_class_enum",
    ]:
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)
