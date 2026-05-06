"""CSV/Excel export builders, profit-margin calc, and download-link generator."""
import base64
import io
from datetime import datetime

import pandas as pd
import streamlit as st
from openpyxl import Workbook

from cogs.formatters import format_cost


def calculate_profit_margins(cost_per_box, sales_price_per_box):
    if sales_price_per_box <= 0:
        return {"profit_per_box": 0, "profit_margin_percent": 0, "roi_percent": 0}

    profit_per_box = sales_price_per_box - cost_per_box
    profit_margin_percent = (profit_per_box / sales_price_per_box) * 100 if sales_price_per_box > 0 else 0
    roi_percent = (profit_per_box / cost_per_box) * 100 if cost_per_box > 0 else 0

    return {
        "profit_per_box": profit_per_box,
        "profit_margin_percent": profit_margin_percent,
        "roi_percent": roi_percent,
    }


def create_csv_export(summary_data_dict, profit_data, calculation_details):
    display_currency = st.session_state.get("display_currency", "USD")
    export_data = []

    export_data.append(["Calculation Details"])
    export_data.append(["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    export_data.append(["Product", calculation_details.get('product', 'N/A')])
    export_data.append(["Quantity", calculation_details.get('quantity', 'N/A')])
    export_data.append(["Shipment Type", calculation_details.get('shipment_type', 'N/A')])
    export_data.append(["Fixed Cost Mode", calculation_details.get('fixed_cost_mode', 'N/A')])
    export_data.append([])

    export_data.append(["Cost Breakdown"])
    for key, value in summary_data_dict.items():
        if value is not None:
            export_data.append([key, value])
    export_data.append([])

    if profit_data:
        export_data.append(["Profit Analysis"])
        export_data.append(["Cost per Box (USD)", f"${profit_data.get('final_cost_per_box_usd', 0):,.2f}"])
        export_data.append([f"Cost per Box ({display_currency})", format_cost(profit_data.get('final_cost_per_box_usd', 0))])

    return pd.DataFrame(export_data)


def create_excel_export(summary_data_dict, profit_data, calculation_details, sensitivity_data):
    display_currency = st.session_state.get("display_currency", "USD")
    wb = Workbook()

    if wb.active is not None:
        wb.remove(wb.active)

    ws_summary = wb.create_sheet("Cost Summary")
    ws_summary.append(["Cost Calculator - Summary Report"])
    ws_summary.append([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    ws_summary.append([])

    ws_summary.append(["Calculation Details"])
    ws_summary.append(["Product", calculation_details.get('product', 'N/A')])
    ws_summary.append(["Quantity", calculation_details.get('quantity', 'N/A')])
    ws_summary.append(["Shipment Type", calculation_details.get('shipment_type', 'N/A')])
    ws_summary.append([])

    ws_summary.append(["Cost Breakdown"])
    for key, value in summary_data_dict.items():
        if value is not None:
            ws_summary.append([key, value])

    if profit_data:
        ws_profit = wb.create_sheet("Profit Analysis")
        ws_profit.append(["Profit Analysis"])
        ws_profit.append([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        ws_profit.append([])
        ws_profit.append(["Cost per Box (USD)", f"${profit_data.get('final_cost_per_box_usd', 0):,.2f}"])
        ws_profit.append([f"Cost per Box ({display_currency})", format_cost(profit_data.get('final_cost_per_box_usd', 0))])

        if sensitivity_data:
            ws_profit.append([])
            ws_profit.append(["Sensitivity Analysis"])
            headers = list(sensitivity_data[0].keys())
            ws_profit.append(headers)
            for row in sensitivity_data:
                ws_profit.append([row[header] for header in headers])

    return wb


def get_download_link(data, filename, file_type):
    if file_type == "csv":
        csv = data.to_csv(index=False, header=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    elif file_type == "excel":
        buffer = io.BytesIO()
        data.save(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel</a>'
    return ""
