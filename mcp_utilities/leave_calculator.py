#!/usr/bin/env python3
"""Leave Calculator FastAPI service."""

import os
from datetime import date, datetime, timedelta
from typing import Dict

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


class LeaveRequest(BaseModel):
    employee_id: int = Field(..., ge=1)
    start_date: date
    end_date: date
    leave_type: str = Field(..., min_length=1)

    @validator("end_date")
    def _validate_date_range(cls, v: date, values: Dict[str, date]):
        start = values.get("start_date")
        if start and v < start:
            raise ValueError("end_date cannot be before start_date")
        return v

    @validator("leave_type")
    def _normalize_leave_type(cls, v: str) -> str:
        return v.strip()


app = FastAPI(title="Leave Calculator API", version="1.0.0")


def get_database_connection():
    """Establish and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_employee_leave_balance(employee_id: int):
    """Fetch leave balances for the given employee."""
    conn = None
    cursor = None
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        query = (
            "SELECT sickleavehours, vacationhours "
            "FROM humanresources.employee "
            "WHERE businessentityid = %s"
        )
        cursor.execute(query, (employee_id,))
        result = cursor.fetchone()
        if result:
            return {
                "sick_leave_hours": result[0],
                "vacation_hours": result[1],
            }
        return None
    except Exception as exc:
        raise RuntimeError(f"Error fetching employee data: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def count_weekdays(start_date: date, end_date: date) -> int:
    """Count weekdays (inclusive) between two dates."""
    current = start_date
    weekday_count = 0
    while current <= end_date:
        if current.weekday() < 5:
            weekday_count += 1
        current += timedelta(days=1)
    return weekday_count


def calculate_leave_hours(start_date: date, end_date: date) -> int:
    """Calculate leave hours based on weekdays only."""
    return count_weekdays(start_date, end_date) * 8


def validate_leave_request(leave_balance, leave_type: str, required_hours: int) -> Dict[str, float]:
    """Validate leave request and return balance information."""
    normalized = leave_type.lower()
    if normalized == "sick" or normalized == "sick day":
        normalized = "sick leave"

    if normalized == "sick leave":
        available_hours = leave_balance["sick_leave_hours"]
        category = "Sick Leave"
    elif normalized in {"annual leave", "annual", "flexi leave", "flexi", "unpaid leave", "unpaid"}:
        available_hours = leave_balance["vacation_hours"]
        category = "Vacation"
    else:
        raise ValueError(f"Invalid leave type: {leave_type}")

    result: Dict[str, float] = {
        "leave_type": normalized,
        "leave_category": category,
        "available_hours": float(available_hours),
    }

    if required_hours <= available_hours:
        remaining_hours = available_hours - required_hours
        result.update(
            {
                "eligible": True,
                "remaining_hours": float(remaining_hours),
                "message": (
                    f"Eligible for leave. {category} balance after leave: "
                    f"{remaining_hours} hours ({remaining_hours / 8:.1f} days)."
                ),
            }
        )
    else:
        shortage = required_hours - available_hours
        result.update(
            {
                "eligible": False,
                "shortage_hours": float(shortage),
                "message": (
                    "Insufficient leave balance. "
                    f"Required: {required_hours} hours ({required_hours / 8:.1f} days), "
                    f"Available: {available_hours} hours ({available_hours / 8:.1f} days), "
                    f"Shortage: {shortage} hours ({shortage / 8:.1f} days)."
                ),
            }
        )

    return result


@app.post("/leave/validate")
def validate_leave(request: LeaveRequest):
    leave_balance = None
    try:
        leave_balance = get_employee_leave_balance(request.employee_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not leave_balance:
        raise HTTPException(status_code=404, detail="Employee not found or leave balance unavailable")

    required_hours = calculate_leave_hours(request.start_date, request.end_date)
    weekdays = count_weekdays(request.start_date, request.end_date)

    try:
        validation = validate_leave_request(leave_balance, request.leave_type, required_hours)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response: Dict[str, float] = {
        "employee_id": request.employee_id,
        "leave_type": validation["leave_type"],
        "leave_category": validation["leave_category"],
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "weekdays": weekdays,
        "requested_hours": required_hours,
        "requested_days": required_hours / 8,
        "required_hours": required_hours,
        "available_hours": validation["available_hours"],
        "eligible": validation["eligible"],
        "message": validation["message"],
        "sick_leave_hours": float(leave_balance.get("sick_leave_hours", 0.0)),
        "vacation_hours": float(leave_balance.get("vacation_hours", 0.0)),
    }

    if validation["eligible"]:
        remaining_hours = validation.get("remaining_hours", 0.0)
        response["remaining_hours"] = remaining_hours
        response["remaining_days"] = remaining_hours / 8
    else:
        shortage_hours = validation.get("shortage_hours", 0.0)
        response["shortage_hours"] = shortage_hours
        response["shortage_days"] = shortage_hours / 8

    return response


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mcp_utilities.leave_calculator:app", host="0.0.0.0", port=8000, reload=False)
