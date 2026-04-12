"""Price alert engine for OOTP PT Optimizer."""
import sqlite3
from datetime import datetime

from app.core.database import get_connection


def create_alert(card_id, alert_type, target_price, conn=None):
    """Create a new price alert.

    Args:
        card_id: The card to watch.
        alert_type: 'below' or 'above'.
        target_price: The target price threshold (PP).
        conn: Optional sqlite3 connection.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    conn.execute(
        "INSERT INTO price_alerts (card_id, alert_type, target_price) VALUES (?, ?, ?)",
        (card_id, alert_type, target_price),
    )
    conn.commit()

    if close_conn:
        conn.close()


def get_active_alerts(conn=None):
    """Get all active, untriggered alerts with card info."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT pa.id, pa.card_id, pa.alert_type, pa.target_price, pa.created_at,
               c.card_title, c.last_10_price
        FROM price_alerts pa
        JOIN cards c ON pa.card_id = c.card_id
        WHERE pa.active = 1 AND pa.triggered = 0
        ORDER BY pa.created_at DESC
    """).fetchall()

    results = [dict(r) for r in rows]

    if close_conn:
        conn.close()

    return results


def check_alerts(conn=None):
    """Check all active alerts against current prices.

    Returns list of newly triggered alerts.
    Marks triggered alerts in the database.
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    active = conn.execute("""
        SELECT pa.id, pa.card_id, pa.alert_type, pa.target_price,
               c.card_title, c.last_10_price
        FROM price_alerts pa
        JOIN cards c ON pa.card_id = c.card_id
        WHERE pa.active = 1 AND pa.triggered = 0
    """).fetchall()

    newly_triggered = []
    for alert in active:
        current_price = alert['last_10_price']
        if current_price is None:
            continue

        triggered = False
        if alert['alert_type'] == 'below' and current_price <= alert['target_price']:
            triggered = True
        elif alert['alert_type'] == 'above' and current_price >= alert['target_price']:
            triggered = True

        if triggered:
            conn.execute(
                "UPDATE price_alerts SET triggered = 1, triggered_at = CURRENT_TIMESTAMP WHERE id = ?",
                (alert['id'],),
            )
            newly_triggered.append({
                'id': alert['id'],
                'card_id': alert['card_id'],
                'card_title': alert['card_title'],
                'alert_type': alert['alert_type'],
                'target_price': alert['target_price'],
                'current_price': current_price,
            })

    if newly_triggered:
        conn.commit()

    if close_conn:
        conn.close()

    return newly_triggered


def dismiss_alert(alert_id, conn=None):
    """Deactivate an alert."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    conn.execute(
        "UPDATE price_alerts SET active = 0 WHERE id = ?",
        (alert_id,),
    )
    conn.commit()

    if close_conn:
        conn.close()


def get_triggered_alerts(conn=None):
    """Get recently triggered alerts (last 7 days)."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    rows = conn.execute("""
        SELECT pa.id, pa.card_id, pa.alert_type, pa.target_price,
               pa.triggered_at, c.card_title, c.last_10_price as current_price
        FROM price_alerts pa
        JOIN cards c ON pa.card_id = c.card_id
        WHERE pa.triggered = 1 AND pa.active = 1
          AND pa.triggered_at >= datetime('now', '-7 days')
        ORDER BY pa.triggered_at DESC
    """).fetchall()

    results = [dict(r) for r in rows]

    if close_conn:
        conn.close()

    return results
